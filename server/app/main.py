"""HTTP routes and application startup; persistence stays behind the repository."""

from contextlib import asynccontextmanager
import logging
import sqlite3

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .models import Acknowledgement, BatchResponse, References, Species, Submission
from .repository import AppError
from .services import AnnotationService
from .settings import Settings
from .sqlite_repository import SQLiteRepository


def create_app(settings=None, repository=None):
    """Build an application, allowing tests to supply isolated settings/storage."""
    settings = settings or Settings()
    repository = repository or SQLiteRepository(settings.local_dir)

    @asynccontextmanager
    async def lifespan(app):
        # Initialize storage on startup, not as a side effect of importing routes.
        repository.initialize()
        app.state.service = AnnotationService(repository, settings)
        yield

    app = FastAPI(title='OSA local annotation server', version='0.5.0', lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.origins),
        allow_methods=['GET', 'POST'],
        allow_headers=['Content-Type'],
    )

    @app.exception_handler(AppError)
    async def application_error(request, exc):
        return JSONResponse(
            status_code=exc.status,
            content={'error': {'message': exc.message}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(
            status_code=422,
            content={
                'error': {
                    'message': 'Invalid request',
                    'details': [
                        {'location': list(e['loc']), 'message': e['msg']}
                        for e in exc.errors()
                    ],
                }
            },
        )

    @app.exception_handler(sqlite3.Error)
    async def storage_error(request, exc):
        logging.getLogger(__name__).exception('Annotation storage failure')
        return JSONResponse(
            status_code=503,
            content={
                'error': {
                    'message': 'Storage unavailable; retry with the same submission ID'
                }
            },
        )

    @app.get('/health')
    def health():
        repository.health()
        return {'status': 'ok', 'storage': 'sqlite', 'schemaVersion': 5}

    @app.get('/api/species', response_model=list[Species])
    def species(request: Request):
        return list(request.app.state.service.species.values())

    @app.get('/api/species/{species_id}/references', response_model=References)
    def references(species_id: str, request: Request):
        service = request.app.state.service
        target = service.require_species(species_id)
        return {
            'target': target.model_dump() | {'clips': []},
            'comparisons': [
                s.model_dump() | {'clips': []}
                for s in service.species.values()
                if s.id != species_id
            ],
        }

    @app.get('/api/species/{species_id}/candidates', response_model=BatchResponse)
    def candidates(species_id: str, request: Request):
        batch = request.app.state.service.get_batch(species_id)
        # Each window uses its full recording URL with explicit playback bounds.
        return {
            'batchId': batch['batchId'],
            'clips': [
                {k: v for k, v in r.items() if k != 'mediaRecordingId'}
                | {'audioUrl': str(request.url_for('recording_audio', recording_id=r['mediaRecordingId'])),
                   'startSeconds': r['offsets'][0], 'endSeconds': r['offsets'][1]}
                for r in batch['clips']
            ],
        }

    @app.post('/api/annotations', response_model=Acknowledgement)
    def annotations(submission: Submission, request: Request):
        return request.app.state.service.save(submission)

    @app.get('/api/submissions/{submission_id}', response_model=Acknowledgement)
    def submission_outcome(submission_id: str):
        return repository.get_submission(submission_id, settings.annotator)

    @app.get('/media/recordings/{recording_id}', name='recording_audio')
    def recording_audio(recording_id: str, request: Request):
        path = request.app.state.service.get_recording_audio(recording_id)
        return FileResponse(path, media_type='audio/wav')

    return app
