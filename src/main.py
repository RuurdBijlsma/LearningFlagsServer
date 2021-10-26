import dataclasses
import functools
import time
from typing import Callable, Tuple, Dict

import Levenshtein
import socketio
import unidecode
from aiohttp import web

import facts
from spacingmodel import SpacingModel, Fact, Response

sio = socketio.AsyncServer(cors_allowed_origins='*', logger=True)
app = web.Application()
sio.attach(app)


def initialize_model(enable_propagation: bool) -> SpacingModel:
    model = SpacingModel(enable_propagation=enable_propagation)
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

    def clean_answer(v: str) -> str:
        return unidecode.unidecode(v.lower()).replace('.', '')

    allowed_mistakes = len(fact.answer) // 7
    correct = Levenshtein.distance(clean_answer(fact.answer), clean_answer(answer)) <= allowed_mistakes

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
@with_model
async def get_stats(*_args, model: SpacingModel, **_kwargs) -> Dict[str, Tuple[str, float]]:
    result = {}

    for fact in model.facts:
        activation = model.calculate_activation(time.time(), fact)

        rof, _ = model.calculate_alpha(time.time(), fact)

        # Sending breaks when activation = -inf, so we'll turn it all to strings
        activation = str(activation)
        result[fact.question] = (activation, rof)

    return result


# subset id is 0 or 1 (or -1 for all flags?)
@sio.event
async def reset_model(sid: str, subset_id: int, enable_propagation: bool, **_kwargs) -> int:
    async with sio.session(sid) as session:
        model = initialize_model(enable_propagation)
        session['model'] = model

        return len(model.facts)


@sio.event
async def get_subset_flags(sid: str):
    pass  # return list of flags with country code and name for the subset
    # [{code: 'SR', name: 'Serbia'}, {code: 'NL', name: 'Netherlands'}]


@sio.event
async def connect(sid, _environ):
    print('connect ', sid)


@sio.event
def disconnect(sid):
    print('disconnect ', sid)


if __name__ == '__main__':
    web.run_app(app, port=5000)
