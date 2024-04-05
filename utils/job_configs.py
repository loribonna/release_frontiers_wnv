def set_params(params, mode=('graph', 'temporal', 'base')):
    assert mode in ('graph', 'temporal', 'base')

    # defaults
    params.temporal_buffer = 0
    params.num_workers = 4
    params.drop_rate = 0.2
    params.batch_size = 32
    params.num_epochs = 30
    params.seed = 42
    params.n_split = 5
    params.scheduler_milestones = [10, 30]
    params.momentum = 0.9

    if mode == 'graph':
        params.bands = [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
        params.colorization = 1

        params.num_multi_images = 0

        params.lr = 0.001
        params.scheduler_type = "step"
        params.scheduler_step = 25

        params.log = "log"
        params.save_params = "params.json"
    elif mode == 'base':
        params.bands = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        params.colorization = 1

        params.num_multi_images = 1

        params.lr = 0.01
        params.scheduler_type = "multi"
        params.scheduler_step = 10

        params.log = "log"
        params.save_params = "params.json"
    elif mode == 'temporal':
        params.bands = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        params.colorization = 0

        params.num_multi_images = 10

        params.lr = 0.01
        params.scheduler_type = "multi"
        params.scheduler_step = 10

    params.log_dir = mode
    params.log = "log"
    params.save_params = "params.json"
    params.test_step = 5
    params.in_channels = sum(params.bands)
    return params
