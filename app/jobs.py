from multiprocessing.managers import SyncManager
from pydantic import BaseModel, Field
from typing import Dict
from uuid import UUID, uuid4


class Job(BaseModel):
    uid: UUID = Field(default_factory=uuid4)
    status: str = "in_progress"
    progress: int = 0
    result: int = None

    def save(self):
        _save_job(self)


_jobs: Dict[UUID, Job] = {}


def _get_jobs():
    return _jobs


_manager = SyncManager(('', 9999), b'password')


def connect():
    _manager.register('jobs')
    _manager.connect()


def _save_job(job):
    _manager.jobs().update([(job.uid, job)])


def find(uid):
    return _manager.jobs().get(uid)


def list_jobs():
    return _manager.jobs().keys()


if __name__ == '__main__':
    _manager.register('jobs', _get_jobs)
    print('serving...')
    server = _manager.get_server()
    server.serve_forever()
