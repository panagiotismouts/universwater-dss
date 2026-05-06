"""
MongoDB serialization helpers.

Handles the impedance mismatch between:
  - Pydantic models that represent DSS documents
  - BSON documents returned by motor

Rules applied consistently:
  1. On WRITE: call model.model_dump(by_alias=True) to get a plain dict.
     The "_id" alias on MongoDocument is handled automatically.
     If the model has no _id (new document), the "_id" key is absent from
     the dict and MongoDB generates an ObjectId.

  2. On READ: motor returns a dict with "_id" as a bson.ObjectId.
     Before passing to Pydantic, coerce all ObjectId values to 24-char
     hex strings using coerce_object_ids().  This keeps the schema layer
     free of any bson dependency.

  3. datetime fields: Motor returns Python datetime objects.  Pydantic
     accepts them directly.  No conversion needed.

  4. None fields: model_dump() includes None-valued fields by default
     (exclude_none=False).  This is deliberate — nullable blueprint fields
     stored as None must be present in the document so MongoDB queries on
     them (e.g. {token_hash: None}) work correctly.
"""

from __future__ import annotations

from typing import Any, Type, TypeVar

from bson import ObjectId
from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def to_document(model: BaseModel) -> dict[str, Any]:
    """
    Serialise a Pydantic model to a MongoDB-ready dict.

    Uses by_alias=True so MongoDocument._id is rendered as "_id".
    The "_id" key is omitted from the result when the model's id field
    is None (new document — let MongoDB generate the ObjectId).

    Returns a plain dict suitable for motor insert_one / update_one.
    """
    doc = model.model_dump(by_alias=True, mode="python")
    # Remove _id=None so MongoDB generates it; keep _id if it is already set
    if doc.get("_id") is None:
        doc.pop("_id", None)
    return doc


def coerce_object_ids(doc: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively coerce bson.ObjectId values to 24-char hex strings in a
    MongoDB result dict.

    Motor returns ObjectId instances for _id and any ObjectId-typed field.
    This function walks the dict (one level + nested dicts/lists) and converts
    all ObjectId values so that Pydantic models receive plain strings.

    Returns a new dict (shallow copy at the top level, deep coercion in place
    for nested structures).
    """
    result: dict[str, Any] = {}
    for key, value in doc.items():
        result[key] = _coerce_value(value)
    return result


def _coerce_value(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dict):
        return {k: _coerce_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_value(item) for item in value]
    return value


def doc_to_model(raw: dict[str, Any], model_cls: Type[M]) -> M:
    """
    Convert a raw motor result dict into a typed Pydantic model instance.

    Steps:
      1. Coerce all ObjectId values to strings
      2. Construct the model using model_validate (supports aliases)

    Usage:
        raw = await col.find_one({"model_id": "water_xgboost_..."})
        model = doc_to_model(raw, ModelRegistryDocument)
    """
    coerced = coerce_object_ids(raw)
    return model_cls.model_validate(coerced)
