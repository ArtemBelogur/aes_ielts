# backend/app/users/routes.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db import schemas, crud, database
from app.auth.dependencies import get_current_user
from app.db.models import User
from typing import List

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/me", response_model=schemas.UserOut)
def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user


@router.get("/", response_model=List[schemas.UserOut])
def read_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return crud.get_users(db, skip=skip, limit=limit)


@router.get("/{user_id}", response_model=schemas.UserOut)
def read_user_by_id(
    user_id: int,
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin" and current_user.id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    db_user = crud.get_user_by_id(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user
