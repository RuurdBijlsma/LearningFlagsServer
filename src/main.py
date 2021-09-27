from aiohttp import web
import socketio

sio = socketio.AsyncServer(cors_allowed_origins='*', logger=True)
app = web.Application()
sio.attach(app)


@sio.event
async def hello(sid, data):
    print("hello message ", data)


@sio.event
def connect(sid, environ):
    print("connect ", sid)


@sio.event
def disconnect(sid):
    print('disconnect ', sid)


if __name__ == '__main__':
    web.run_app(app, port=5000)
