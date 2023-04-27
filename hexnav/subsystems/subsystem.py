import click


class Subsystem:
    def __init__(self):
        pass

    def available(self):
        return False


class SubsystemManager:
    def __init__(self):
        self._subsystems = {}

    def register(self, name, subsystem, mandatory=True):
        if not subsystem.available() and mandatory:
            click.secho(f'Error: Subsystem "{name}" not available.', fg='red')
            exit()
        self._subsystems[name] = subsystem

    def get(self, name, default=None):
        return self._subsystems.get(name, default)

    def show(self):
        click.secho('Available subsystems:')
        for name, subsystem in self._subsystems.items():
            status = '✅' if subsystem.available() else '❌'
            click.secho(f'  {status} {name}')