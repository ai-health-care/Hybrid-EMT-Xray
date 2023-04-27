#!/usr/bin/env python3

"""
IGTLink dummy server.
You can use this to test the orchestration without turning on the EMT.
All it does is sending 4x4 identity matrices.
"""

import time

# 3rd party
import click
import numpy as np
import pyigtl


@click.command()
@click.option('--port', '-p', default=18944)
def dummyserver(port):
    server = pyigtl.OpenIGTLinkServer(port=port, local_server=True)
    click.secho('IGTLink server initialized.', fg='green')
    while True:
        if not server.is_connected():
            # Wait for client to connect
            time.sleep(0.1)
            continue

        matrix = np.eye(4)
        transform_message = pyigtl.TransformMessage(
            matrix,
            device_name='Tracker'
        )
        server.send_message(transform_message)


if __name__ == '__main__':
    dummyserver()