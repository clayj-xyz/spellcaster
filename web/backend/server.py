from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from spellcaster_manager import SpellcasterManager
from spellcaster_viewer import SpellcasterViewer

spellcaster_manager = None
spellcaster_viewer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global spellcaster_manager, spellcaster_viewer

    spellcaster_manager = SpellcasterManager()
    spellcaster_manager.debug()
    spellcaster_viewer = SpellcasterViewer()
    yield
    del spellcaster_viewer
    del spellcaster_manager
    

app = FastAPI(lifespan=lifespan)

@app.get("/stream")
async def stream(request: Request):
    
    return StreamingResponse(
        spellcaster_viewer.get_stream(request),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )
