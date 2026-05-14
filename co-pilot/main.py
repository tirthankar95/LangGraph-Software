from langagents.build_graph import build as graph_build
from fastapi import FastAPI, Depends, HTTPException
from backend.database import get_db, init_db
from contextlib import asynccontextmanager
from sqlalchemy.exc import IntegrityError
from langgraph.types import Command
from sqlalchemy.orm import Session
from backend.models import User
from pydantic import BaseModel
from typing import Literal
from uuid import uuid4


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize application resources at startup."""
    init_db()
    yield


app = FastAPI(lifespan=lifespan)


class ReplyRequest(BaseModel):
    user_id: str
    user_request: str

class ApproveRequest(BaseModel):
    user_id: str
    thread_id: str
    user_request: Literal["approve", "reject"]

class CreateUserRequest(BaseModel):
    user_id: str
    email: str
    name: str | None = None


@app.get("/")
async def root():
    return {"message": "Welcome to CodeGen agent!"}


@app.post("/users", status_code=201)
async def create_user(payload: CreateUserRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.user_id == payload.user_id).first()
    if existing_user:
        raise HTTPException(status_code=409, detail=f"user_id '{payload.user_id}' already exists")

    existing_email = db.query(User).filter(User.email == payload.email).first()
    if existing_email:
        raise HTTPException(status_code=409, detail=f"email '{payload.email}' already exists")

    user = User(user_id=payload.user_id, email=payload.email, name=payload.name)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="User already exists")

    db.refresh(user)
    return {
        "user_id": user.user_id,
        "email": user.email,
        "name": user.name,
    }

'''
The graph instance is stateless. 
It's just a compiled execution template — it holds the node definitions, edges, and a reference to the checkpointer. 
It holds zero per-user data itself.
'''
graph = graph_build()


'''
Checkpoints after END — they are NOT deleted
This is the important one. When a graph reaches END, the final checkpoint is written to Postgres and kept permanently. 
Nothing is cleaned up automatically.
'''
@app.post("/reply")
async def reply(payload: ReplyRequest, db: Session = Depends(get_db)):
    """
    Process a user request after authenticating the user_id.
    Args:
        payload: ReplyRequest containing user_id and user_request
        db: Database session
    Returns:
        Response with user_id and reply
    Raises:
        HTTPException: If user_id is not found in the database
    """
    # Authenticate user_id by checking if it exists in the database
    user = db.query(User).filter(User.user_id == payload.user_id).first()
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail=f"User {payload.user_id} not found or not authenticated"
        )
    config = {"configurable": {"thread_id": f"{payload.user_id}_{uuid4()}"}}
    response = graph.invoke({"user_query": payload.user_request}, config=config)
    print(config)
    return {
        "user_id": payload.user_id,
        "thread_id": config["configurable"]["thread_id"],
        "message": response['__interrupt__'][0].value['message'],
        "plan": response['__interrupt__'][0].value['plan']
    }

@app.post("/approve_reply")
async def handle_approval(payload: ApproveRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.user_id == payload.user_id).first()
    if not user:
        raise HTTPException(
            status_code=401,
            detail=f"User {payload.user_id} not found or not authenticated"
        )
    # graph = graph_build()
    config = {"configurable": {"thread_id": f"{payload.thread_id}"}}
    print(config)
    response = graph.invoke(Command(resume=payload.user_request), config)
    return {
        "user_id": payload.user_id,
        "thread_id": payload.thread_id,
        "message": response['break_plan']
    }