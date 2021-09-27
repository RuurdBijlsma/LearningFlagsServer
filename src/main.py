import numpy as np
import socketio
from aiohttp import web

# create a Socket.IO server
sio = socketio.AsyncServer(cors_allowed_origins='*', logger=True)
app = web.Application()
sio.attach(app)
# app = socketio.WSGIApp(sio)

games = {}


# Events are sent from the web client to this server
@sio.event
async def start_game(sid, arg):
    try:
        print('received start game with arg', arg)
    except Exception as e:
        print(e)  # Events are sent from the web client to this server


@sio.event
def connect(sid, environ):
    print('connect sid:', sid)


@sio.event
def disconnect(sid):
    games.pop(sid)
    print('disconnect sid:', sid)


if __name__ == '__main__':
    web.run_app(app, port=5000)
