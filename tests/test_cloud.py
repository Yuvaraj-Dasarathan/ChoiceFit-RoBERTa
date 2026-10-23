import json
import tarfile
import pytest

from cloud.launch import build_request, package_code


def test_request_schema_and_cost_controls():
    pytest.importorskip('botocore')
    import botocore.session
    from botocore.validate import validate_parameters
    role = 'arn:aws:iam::123456789012:role/SageMakerChoiceFit'
    request = build_request('choicefit-test', 'example-bucket', role, 'choicefit', spot=True)
    shape = botocore.session.get_session().get_service_model('sagemaker').operation_model('CreateTrainingJob').input_shape
    validate_parameters(request, shape)
    assert request['ResourceConfig']['InstanceCount'] == 1
    assert request['ResourceConfig']['InstanceType'] == 'ml.g4dn.xlarge'
    assert request['StoppingCondition'] == {'MaxRuntimeInSeconds': 900, 'MaxWaitTimeInSeconds': 3600}
    assert request['CheckpointConfig']['LocalPath'] == '/opt/ml/checkpoints'
    assert request['EnableManagedSpotTraining'] is True
    assert 'MaxWaitTimeInSeconds' not in build_request('test', 'bucket', role, 'prefix')['StoppingCondition']
    reused = build_request('new-job', 'bucket', role, 'prefix', input_job_name='old-job')
    assert reused['InputDataConfig'][0]['DataSource']['S3DataSource']['S3Uri'].endswith('/old-job/train/')
    assert json.loads(reused['HyperParameters']['sagemaker_submit_directory']).endswith('/old-job/code/source.tar.gz')


def test_code_archive_excludes_data_and_environments(tmp_path):
    archive = tmp_path / 'source.tar.gz'
    package_code(archive)
    with tarfile.open(archive) as source:
        names = source.getnames()
    assert 'entrypoint.py' in names
    assert 'requirements.txt' in names
    assert 'src/train.py' in names and 'src/checkpoint.py' in names
    assert not any(name.startswith(('data/', '.venv/', '.git/', 'artifacts/')) for name in names)


def test_entrypoint_maps_channels_and_boolean_flags(tmp_path, monkeypatch):
    import json
    import sys
    import cloud.entrypoint as entrypoint
    monkeypatch.setattr(sys, 'argv', ['entrypoint.py'])
    monkeypatch.setenv('HF_HUB_OFFLINE', '0')
    monkeypatch.setattr(entrypoint.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setenv('CHOICEFIT_CONFIG', json.dumps({'epochs': 1, 'gradient-checkpointing': True}))
    monkeypatch.setenv('SM_CHANNEL_BASE', '/input/base')
    monkeypatch.setenv('SM_CHANNEL_TEST', str(tmp_path))
    monkeypatch.setenv('SM_MODEL_DIR', str(tmp_path))
    captured = {}
    monkeypatch.setattr(entrypoint, 'main', lambda: captured.update(argv=list(sys.argv)))
    monkeypatch.setattr(entrypoint, 'ChoiceModel', lambda path, device: object())
    monkeypatch.setattr(entrypoint, 'iter_data', lambda path: [])
    monkeypatch.setattr(entrypoint, 'evaluate', lambda *args, **kwargs: {'examples': 16})
    entrypoint.run()
    assert '--gradient-checkpointing' in captured['argv']
    assert 'True' not in captured['argv']
    assert captured['argv'][captured['argv'].index('--model') + 1] == '/input/base'
    assert (tmp_path / 'test_metrics.json').exists()
