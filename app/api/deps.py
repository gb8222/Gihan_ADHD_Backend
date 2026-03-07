from fastapi import Header, HTTPException, status

from app.core.security import decode_supabase_jwt


def _bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    return authorization.split(" ", 1)[1].strip()


async def require_supabase_user(authorization: str | None = Header(default=None)):
    token = _bearer(authorization)
    payload = decode_supabase_jwt(token)
    # Supabase uses 'sub' as user id (uuid)
    return {"token": token, "user_id": payload.get("sub"), "claims": payload}
