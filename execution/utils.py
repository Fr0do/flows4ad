__all__ = [
    'make_drop_last_false_loader',
    'make_stepwise_generator'
]   


def make_drop_last_false_loader(loader):
    # temporarily do not drop last
    template = dict(loader.__dict__)
    # drop attributes that will be auto-initialized
    to_drop = [k for k in template if k.startswith("_") or k == "batch_sampler"]
    for item in to_drop:
        template.pop(item)
    template['drop_last'] = False
    return type(loader)(**template)


def make_stepwise_generator(loader, num_steps: int):
    steps_done = 0
    while True:
        for batch in loader:
            if steps_done == num_steps:
              return
            yield batch
            steps_done += 1