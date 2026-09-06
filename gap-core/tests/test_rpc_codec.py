"""Codec roundtrips for the stdio tool RPC framing.

The wire format is `<4-byte big-endian length><msgpack payload>`. Numpy
arrays round-trip via msgpack_numpy's patched extension types. These
tests pin the byte layout that the gap-side ``ToolClient`` and the
in-bundle ``gap_tool_server`` both rely on.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from gap_core.rpc.codec import FrameError, decode_frame, encode_frame, write_frame


def _roundtrip(payload):
    return decode_frame(io.BytesIO(encode_frame(payload)))


def test_roundtrip_basic_dict():
    payload = {
        "id": "req-1",
        "kind": "call",
        "tool": "sam3.segment",
        "args": {"prompt": "the red mug", "threshold": 0.5},
    }
    assert _roundtrip(payload) == payload


def test_roundtrip_numpy_array():
    payload = {
        "id": "req-2",
        "kind": "result",
        "result": {
            "mask": np.ones((4, 8), dtype=np.uint8),
            "score": 0.93,
            "box": np.array([0.1, 0.2, 0.5, 0.6], dtype=np.float32),
        },
    }
    decoded = _roundtrip(payload)
    assert decoded["id"] == "req-2"
    assert decoded["kind"] == "result"
    assert decoded["result"]["score"] == pytest.approx(0.93)
    np.testing.assert_array_equal(decoded["result"]["mask"], payload["result"]["mask"])
    np.testing.assert_array_equal(decoded["result"]["box"], payload["result"]["box"])
    assert decoded["result"]["mask"].dtype == np.uint8
    assert decoded["result"]["box"].dtype == np.float32


def test_roundtrip_nested_structure():
    payload = {
        "id": "hs-0",
        "kind": "catalog",
        "tools": [
            {
                "name": "geometry.compute_obb",
                "summary": "...",
                "tags": ["geom"],
                "schema": {
                    "inputs": {"points": {"type_str": "ndarray"}},
                    "outputs": {"obb": {"type_str": "OrientedBoundingBox"}},
                },
            },
        ],
    }
    decoded = _roundtrip(payload)
    assert decoded == payload


def test_decode_empty_stream_returns_none():
    """Clean EOF is signaled by None (not an exception) so the server's
    read loop can exit normally when gap closes stdin."""
    assert decode_frame(io.BytesIO(b"")) is None


def test_decode_truncated_header_raises():
    with pytest.raises(FrameError, match="length header"):
        decode_frame(io.BytesIO(b"\x00\x00"))


def test_decode_truncated_body_raises():
    # length says 999 bytes but stream only has a few.
    with pytest.raises(FrameError, match="body"):
        decode_frame(io.BytesIO(b"\x00\x00\x03\xe7" + b"abc"))


def test_decode_rejects_oversized_frame_before_reading_body():
    stream = io.BytesIO(b"\x00\x00\x01\x00")
    with pytest.raises(FrameError, match="exceeds limit 64"):
        decode_frame(stream, max_frame_bytes=64)
    assert stream.tell() == 4


def test_decode_non_mapping_payload_raises():
    """Frames MUST be dicts — a top-level list would mask routing fields."""
    import msgpack

    body = msgpack.packb([1, 2, 3], use_bin_type=True)
    import struct

    framed = struct.pack(">I", len(body)) + body
    with pytest.raises(FrameError, match="must be a mapping"):
        decode_frame(io.BytesIO(framed))


def test_decode_garbage_payload_raises():
    """Bytes that aren't valid msgpack must surface a decode error rather
    than silently returning None."""
    import struct

    framed = struct.pack(">I", 3) + b"\xc1\xc1\xc1"  # \xc1 is reserved
    with pytest.raises(FrameError, match="msgpack decode"):
        decode_frame(io.BytesIO(framed))


def test_write_frame_flushes():
    """The flush is what makes the RPC interactive — without it the server
    would buffer its reply forever."""

    class _Stream(io.BytesIO):
        flushed = 0

        def flush(self):
            self.flushed += 1
            super().flush()

    stream = _Stream()
    written = write_frame(stream, {"id": "x", "kind": "ping"})
    assert stream.flushed >= 1
    assert written == len(stream.getvalue())
    stream.seek(0)
    assert decode_frame(stream) == {"id": "x", "kind": "ping"}


def test_two_frames_in_one_stream():
    """The framing is length-delimited so back-to-back frames decode
    independently — verifies the request/reply loop works on a shared pipe."""
    a = {"id": "1", "kind": "call", "tool": "x", "args": {}}
    b = {"id": "2", "kind": "result", "result": {"ok": True}}
    stream = io.BytesIO(encode_frame(a) + encode_frame(b))
    assert decode_frame(stream) == a
    assert decode_frame(stream) == b
    assert decode_frame(stream) is None  # clean EOF
