"""Build/review a SageMaker request locally. Only --launch uploads or starts a job."""
import argparse
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from src.data import audit_splits
from src.model import iter_data
from src.checkpoint import file_digest
from src.prepare_typed_decisions import fits_token_budget
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
IMAGE_TAG = '2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker'


def build_request(job_name, bucket, role, prefix, spot=False, max_run=900, max_wait=3600,
                  training_config=None):
    region = 'us-east-1'
    location = f's3://{bucket}/{prefix}/{job_name}'
    channels = []
    for channel in ('train', 'validation', 'test', 'base'):
        channels.append({'ChannelName': channel, 'DataSource': {'S3DataSource': {
            'S3DataType': 'S3Prefix', 'S3Uri': f'{location}/{channel}/', 'S3DataDistributionType': 'FullyReplicated'}}})
    request = {
        'TrainingJobName': job_name, 'RoleArn': role,
        'AlgorithmSpecification': {'TrainingImage': f'763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:{IMAGE_TAG}',
                                   'TrainingInputMode': 'File'},
        'InputDataConfig': channels,
        # SageMaker appends <job-name>/output/model.tar.gz itself.
        'OutputDataConfig': {'S3OutputPath': f's3://{bucket}/{prefix}/'},
        'ResourceConfig': {'InstanceType': 'ml.g4dn.xlarge', 'InstanceCount': 1, 'VolumeSizeInGB': 50},
        'StoppingCondition': {'MaxRuntimeInSeconds': max_run},
        'CheckpointConfig': {'S3Uri': location + '/checkpoints/', 'LocalPath': '/opt/ml/checkpoints'},
        'EnableManagedSpotTraining': spot,
        'HyperParameters': {'sagemaker_program': json.dumps('entrypoint.py'),
                            'sagemaker_submit_directory': json.dumps(location + '/code/source.tar.gz'),
                            'sagemaker_region': json.dumps(region)},
        'Environment': {'CHOICEFIT_CONFIG': json.dumps(training_config or {
            'epochs': 1, 'batch-size': 1, 'max-pairs': 4, 'gradient-accumulation-steps': 4,
            'eval-pair-batch-size': 2, 'gradient-checkpointing': True, 'max-length': 192,
            'checkpoint-every': 50, 'log-every': 25}), 'TOKENIZERS_PARALLELISM': 'false'},
    }
    if spot:
        request['StoppingCondition']['MaxWaitTimeInSeconds'] = max_wait
    return request


def preflight(data_dir, base_model):
    data_dir, base_model = Path(data_dir), Path(base_model)
    if not (base_model / 'model.safetensors').is_file():
        raise ValueError('Base model directory must contain model.safetensors; download/prepare it locally first')
    counts = audit_splits(data_dir / 'train.jsonl', data_dir / 'validation.jsonl', max_pairs=4)
    # Test leakage checks also run before any paid resources are requested.
    audit_splits(data_dir / 'train.jsonl', data_dir / 'test.jsonl', max_pairs=4)
    audit_splits(data_dir / 'validation.jsonl', data_dir / 'test.jsonl', max_pairs=4)
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    hashes = {}
    for split in ('train', 'validation', 'test'):
        path = data_dir / f'{split}.jsonl'
        total = 0
        for row in iter_data(path):
            if len(row['choices']) > 4 or not fits_token_budget(row, tokenizer, 192):
                raise ValueError(f'{split}: local pilot requires <=4 choices and complete pairs <=192 tokens')
            total += 1
        counts[split] = total
        hashes[split] = file_digest(path)
    return {'counts': counts, 'sha256': hashes}


def package_code(destination):
    with tarfile.open(destination, 'w:gz') as archive:
        archive.add(ROOT / 'cloud/entrypoint.py', arcname='entrypoint.py')
        archive.add(ROOT / 'cloud/requirements.txt', arcname='requirements.txt')
        for path in sorted((ROOT / 'src').glob('*.py')):
            # Exclude standalone experiments and dataset preparation dependencies.
            if path.name in {'basic-roberta.py', 'prepare_typed_decisions.py', 'prepare_full_typed_decisions.py'}:
                continue
            archive.add(path, arcname='src/' + path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', default='yuvaraj-choicefit-training', help='Existing private S3 bucket in us-east-1')
    parser.add_argument('--role', default='arn:aws:iam::612242785639:role/SageMakerChoiceFit', help='SageMaker execution role ARN')
    parser.add_argument('--base-model', required=True, help='Local RoBERTa snapshot (weights + tokenizer)')
    parser.add_argument('--data-dir', default='data/typed-full')
    parser.add_argument('--prefix', default='choicefit')
    parser.add_argument('--job-name', default='choicefit-pilot-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--spot', action='store_true')
    parser.add_argument('--max-run', type=int, default=86400, help='Maximum billable training seconds')
    parser.add_argument('--max-wait', type=int, default=172800, help='Spot wait plus run limit')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--eval-every', type=int, default=10000, help='Optimizer updates between full validation passes')
    parser.add_argument('--early-stopping-patience', type=int, default=5)
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0001)
    parser.add_argument('--early-stopping-min-validations', type=int, default=3)
    parser.add_argument('--checkpoint-every', type=int, default=5000)
    parser.add_argument('--log-every', type=int, default=10000)
    parser.add_argument('--launch', action='store_true', help='Upload and start ONE paid training job; otherwise dry run')
    args = parser.parse_args()
    if not 60 <= args.max_run <= 604800 or args.max_wait < args.max_run or args.epochs < 1:
        parser.error('Require 60–604800 seconds max-run, max-wait >= max-run, and positive epochs')
    if min(args.learning_rate, args.eval_every, args.early_stopping_patience,
           args.checkpoint_every, args.log_every) <= 0 or args.early_stopping_min_delta < 0 or args.early_stopping_min_validations < 0:
        parser.error('Training intervals and learning rate must be positive; early-stopping deltas/minimum must be zero or positive')
    if not re.fullmatch(r'[a-zA-Z0-9](-*[a-zA-Z0-9])*', args.job_name) or len(args.job_name) > 63:
        parser.error('Invalid SageMaker job name')
    report = preflight(args.data_dir, args.base_model)
    training_config = {
        'epochs': args.epochs, 'batch-size': 1, 'max-pairs': 4, 'gradient-accumulation-steps': 4,
        'eval-pair-batch-size': 16, 'gradient-checkpointing': True, 'max-length': 192,
        'learning-rate': args.learning_rate, 'eval-every': args.eval_every,
        'early-stopping-patience': args.early_stopping_patience,
        'early-stopping-min-delta': args.early_stopping_min_delta,
        'early-stopping-min-validations': args.early_stopping_min_validations,
        'checkpoint-every': args.checkpoint_every, 'log-every': args.log_every,
        'convergence-jsonl': '/opt/ml/model/convergence.jsonl',
    }
    request = build_request(args.job_name, args.bucket, args.role, args.prefix.strip('/'),
                            args.spot, args.max_run, args.max_wait, training_config)
    # Botocore validates request shape without credentials or AWS requests.
    import botocore.session
    from botocore.validate import validate_parameters
    shape = botocore.session.get_session().get_service_model('sagemaker').operation_model('CreateTrainingJob').input_shape
    validate_parameters(request, shape)
    print(json.dumps({'preflight': report, 'request': request, 'launch': args.launch}, indent=2))
    if not args.launch:
        print('DRY RUN: nothing uploaded; no training job created.')
        return
    import boto3
    from botocore.exceptions import ClientError
    session = boto3.Session(region_name='us-east-1')
    s3 = session.client('s3')
    bucket_region = s3.get_bucket_location(Bucket=args.bucket).get('LocationConstraint') or 'us-east-1'
    if bucket_region != 'us-east-1':
        raise ValueError('Bucket must be in us-east-1 for this configuration')
    sm = session.client('sagemaker')
    # Do not silently reuse a previous job/checkpoint location.
    try:
        sm.describe_training_job(TrainingJobName=args.job_name)
    except ClientError as exc:
        if exc.response['Error']['Code'] != 'ValidationException':
            raise
    else:
        raise ValueError('Job name already exists; choose a new job name')
    key = f"{args.prefix.strip('/')}/{args.job_name}"
    if s3.list_objects_v2(Bucket=args.bucket, Prefix=key + "/", MaxKeys=1).get("KeyCount", 0):
        raise ValueError("S3 job prefix already contains objects; use a new job name")
    import tempfile
    with tempfile.TemporaryDirectory(prefix='choicefit-code-') as directory:
        bundle = Path(directory) / 'source.tar.gz'
        package_code(bundle)
        s3.upload_file(str(bundle), args.bucket, key + '/code/source.tar.gz')
    for split in ('train', 'validation', 'test'):
        s3.upload_file(str(Path(args.data_dir) / f'{split}.jsonl'), args.bucket, f'{key}/{split}/{split}.jsonl')
    # Allowlist: no .git, environments, unrelated datasets, or private config.
    for name in ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
                 'special_tokens_map.json', 'vocab.json', 'merges.txt'):
        path = Path(args.base_model) / name
        if path.is_file():
            s3.upload_file(str(path), args.bucket, f'{key}/base/{name}')
    print(json.dumps(sm.create_training_job(**request), default=str))
    print(f'Monitor: aws sagemaker describe-training-job --region us-east-1 --training-job-name {args.job_name}')
    print(f'Stop: aws sagemaker stop-training-job --region us-east-1 --training-job-name {args.job_name}')


if __name__ == '__main__':
    main()
