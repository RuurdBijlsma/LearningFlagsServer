import dataclasses
import functools
import time
from typing import Callable, Tuple

import socketio
from aiohttp import web

import facts
from spacingmodel import SpacingModel, Fact, Response

sio = socketio.AsyncServer(cors_allowed_origins='*', logger=True)
app = web.Application()
sio.attach(app)


def initialize_model() -> SpacingModel:
    model = SpacingModel()
    facts.load_facts(model)
    return model


def get_fact_by_country_code(model: SpacingModel, country_code: str) -> Fact:
    return next(fact for fact in model.facts if fact.question == country_code)


def with_model(f: Callable) -> Callable:
    @functools.wraps(f)
    async def wrapper(sid: str, *args, **kwargs):
        async with sio.session(sid) as session:
            model = session['model']
            return await f(*args, sid=sid, **kwargs, model=model)

    return wrapper


@sio.event
@with_model
async def next_fact(*_args, model: SpacingModel, **_kwargs) -> Tuple[Fact, bool]:
    fact, is_new = model.get_next_fact(time.time())

    return dataclasses.asdict(fact), is_new


@sio.event
@with_model
async def register_response(country_code: str, answer: str, rt: float, model: SpacingModel, **_kwargs) -> bool:
    fact = get_fact_by_country_code(model, country_code)

    correct = fact.answer.lower() == answer.lower()

    now = time.time()
    # TODO do we need to do this?
    start_time = now - rt

    response = Response(
        fact=fact,
        start_time=start_time,
        rt=rt,
        correct=correct,
    )

    model.register_response(response)

    return correct


@sio.event
async def connect(sid, _environ):
    async with sio.session(sid) as session:
        session['model'] = initialize_model()


@sio.event
def disconnect(sid):
    print('disconnect ', sid)


if __name__ == '__main__':
    web.run_app(app, port=5000)
