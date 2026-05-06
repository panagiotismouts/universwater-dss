"""
Shared base models and type helpers for the DSS schema layer.

Provides:
  DSSBaseModel     — common Pydantic config for all DSS models
  PyObjectId       — str alias for MongoDB ObjectId fields
  MongoDocument    — base for persistence models that may carry an _id
  utc_now()        — canonical UTC datetime factory for default_factory use
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# UTC timestamp helper
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# PyObjectId — string alias for MongoDB ObjectId values
#
# MongoDB ObjectIds are serialised as 24-character hex strings when crossing
# the Python↔MongoDB boundary via motor.  We represent them as plain str in
# Pydantic models so:
#   - no motor/bson import is required in the schema layer
#   - JSON serialization is zero-cost
#   - FK fields are self-documenting by type
# ---------------------------------------------------------------------------

PyObjectId = Annotated[str, Field(min_length=24, max_length=24)]


# ---------------------------------------------------------------------------
# DSSBaseModel — common config for all DSS Pydantic models
# ---------------------------------------------------------------------------

class DSSBaseModel(BaseModel):
    """
    Base class for all DSS Pydantic models.

    Configuration:
      - populate_by_name=True  allows both alias and field name to be used
        during construction, which simplifies tests and script usage.
      - use_enum_values=True   stores the .value of Enum fields, keeping
        serialized output as plain strings/ints.
      - frozen=False           documents are mutable within Python; immutability
        is enforced at the MongoDB layer by application convention, not here.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        frozen=False,
    )


# ---------------------------------------------------------------------------
# MongoDocument — base for documents that may carry a MongoDB _id
# ---------------------------------------------------------------------------

class MongoDocument(DSSBaseModel):
    """
    Optional _id field for models that represent MongoDB documents.

    When reading from MongoDB, motor returns _id as an ObjectId which should
    be coerced to str before constructing this model.  When writing a new
    document, omit _id and let MongoDB generate it.

    The field is aliased as "_id" so that motor result dicts map cleanly:
        doc = MyDocument.model_validate(mongo_result_dict)
    """

    id: Optional[PyObjectId] = Field(default=None, alias="_id")
