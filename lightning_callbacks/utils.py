_CALLBACKS = {}
def register_callback(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CALLBACKS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CALLBACKS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_callback_by_name(name):
    return _CALLBACKS[name]

def get_callbacks(config):
    callbacks=[get_callback_by_name('ema')(decay=config.model.ema_rate)]

    if config.logging.top_k is not None:
      callbacks.append(get_callback_by_name('CheckpointTopK')(config))
    if config.logging.every_n_epochs is not None:
      callbacks.append(get_callback_by_name('CheckpointEveryNepochs')(config))
    if config.logging.envery_timedelta is not None:
      callbacks.append(get_callback_by_name('CheckpointTime')(config))
    

    if config.eval.callback is not None:
      callbacks.append(get_callback_by_name(config.eval.callback)(show_evolution=False, 
                                                                  eval_config=config.eval, 
                                                                  data_config=config.data,
                                                                  approach = config.training.conditioning_approach))
    if config.training.visualization_callback is not None:
      if isinstance(config.training.visualization_callback, list):
        for callback in config.training.visualization_callback:
          callbacks.append(get_callback_by_name(callback)(show_evolution=config.training.show_evolution))
      else:
        callbacks.append(get_callback_by_name(config.training.visualization_callback)(show_evolution=config.training.show_evolution))
    if config.training.lightning_module in ['conditional_decreasing_variance','haar_conditional_decreasing_variance'] :
      callbacks.append(get_callback_by_name('decreasing_variance_configuration')(config))
    else:
      callbacks.append(get_callback_by_name('configuration')())

    return callbacks


