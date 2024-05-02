from typing import Annotated, List, Union

from fastapi import APIRouter, File, UploadFile, Request, Header, HTTPException
from fastapi.background import BackgroundTasks

from app.core.database import SessionLocal

from app.utils.utils import (
    save_audio_file,
    transcribe_file,
    get_audio_duration,
    get_model_name,
)
from app.core.models import AuthTokenController, TranscribeController
from app.api.models import Transcription

router = APIRouter()
database = SessionLocal()


@router.post("/", response_model=Transcription)
async def post_audio(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    model: str = "medium",
    Authentication: Annotated[Union[str, None], Header()] = None,
):
    try:
        userId = AuthTokenController(database).get_userid_from_token(Authentication)
        file_path = save_audio_file(file)
        [data, output_audio_path] = transcribe_file(file_path, get_model_name(model))
        background_tasks.add_task(
            create_transcribe_record, database, userId, data, output_audio_path
        )

        return Transcription(filename=file.filename, text=data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=exc.__str__())


def create_transcribe_record(database, userId, data, file_path):
    duration = get_audio_duration(file_path)
    TranscribeController(database).create(user_id=userId, text=data, duration=duration)
