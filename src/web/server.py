from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, HTMLResponse

from web.spellcaster_manager import SpellcasterManager
from web.spellcaster_viewer import SpellcasterViewer

spellcaster_manager = None
spellcaster_viewer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global spellcaster_manager, spellcaster_viewer

    spellcaster_manager = SpellcasterManager()
    spellcaster_viewer = SpellcasterViewer()
    yield
    del spellcaster_viewer
    del spellcaster_manager
    

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse('index.html', {"request": request, "spellcaster_mode": spellcaster_manager.mode})


@app.get("/mode")
def get_mode():
    return spellcaster_manager.mode


@app.get("/activate")
def activate():
    spellcaster_manager.change_mode("active")
    return HTMLResponse(
        """
        <button hx-get="/deactivate" hx-target="this" hx-swap="outerHTML" class="btn primary">
            Deactivate
        </button>
        """
    )


@app.get("/deactivate")
def deactivate():
    spellcaster_manager.change_mode("standby")
    return HTMLResponse(
        """
        <button hx-get="/activate" hx-target="this" hx-swap="outerHTML" class="btn primary">
            Activate
        </button>
        """
    )


@app.get("/stream")
async def stream(request: Request):
    if spellcaster_manager.mode == "standby":
        return "error"
    
    return StreamingResponse(
        spellcaster_viewer.get_stream(request),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )
