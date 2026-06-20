"""LLM-driven shot breakdown: turns a composed scene (the reader's
highlighted span, already resolved to character/location state by
app/scene_composer.py) into a video plan -- a fixed "world" of identity
anchors plus one or more cinematic text-to-video diffusion shots, mirroring
a hand-authored script of the shape:

    WORLD = {"hare": "...", "tortoise": "...", "world": "...", "look": "..."}
    SCENES = [("01_the_setup", build_prompt(camera=..., action=..., light=...)), ...]
    NEG = "..."

A short highlighted moment becomes a single shot. A longer span covering a
location change, a time jump, or a sequence of distinct physical actions
becomes several, meant to be generated as separate clips and stitched
together in order.

`world` and `negative_prompt` are built deterministically from the DB's
Tier 1/2 state -- the full character/location visual descriptions are fixed
identity anchors, spliced verbatim into every shot's `prompt`. The LLM is
asked for ONLY the camera, action, and lighting per shot, never the
character/setting prose -- letting it re-describe characters per shot is
exactly what caused identity drift (a full bio in shot 1, a fragment in
shot 2, nothing in shot 3) in earlier iterations of this module. Fixing the
anchors and varying only what actually changes per shot is what keeps a
multi-shot sequence visually continuous.

Uses the same structured-decoding pattern as ingestion (ingestion/llm_client.py,
ingestion/schemas.py): the LLM is constrained to a Pydantic schema via vLLM's
structured outputs, so the caller only has to handle truncation and
schema-validation failures, never malformed JSON.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import settings
from app.schemas import ComposedScenePayload, VideoPlanPayload, VideoShotPayload, VideoWorldPayload
from ingestion.llm_client import GpuWorkerPool

SYSTEM_PROMPT = """You are a cinematographer breaking a book scene down into \
shots for a text-to-video diffusion model.

Given the resolved story state for a reader's highlighted passage, decide \
whether it needs ONE shot or SEVERAL sequential shots that a video pipeline \
will stitch together into one clip. Use multiple shots only when the \
passage genuinely covers distinct beats -- a location change, a time jump, \
or a sequence of discrete physical actions. A single static moment should \
stay one shot. Never plan more shots than there are distinct beats.

For each shot, write THREE separate fields:
- camera: the shot type and camera angle/movement only, e.g. "Cinematic wide \
low-angle establishing shot, the camera slowly pushing in."
- action: the precise physical action or dialogue beat happening in THIS \
shot only -- who does what, where they are relative to each other and the \
frame. Do not redescribe what any character or the setting looks like; that \
is supplied separately and must not be repeated or contradicted here.
- light: the lighting and time of day for this shot only, e.g. "Soft dawn \
golden-hour light."

Never mention paragraph numbers, character IDs, or any other database \
bookkeeping. Give each shot a short slug id like "01_the_chase" that \
reflects its place in sequence."""


class ShotCandidate(BaseModel):
    shot_id: str = Field(..., description="Short slug reflecting sequence order, e.g. '01_the_chase'")
    camera: str = Field(..., description="Shot type and camera angle/movement only")
    action: str = Field(..., description="The physical action/dialogue beat in this shot only -- no character or setting description")
    light: str = Field(..., description="Lighting and time of day for this shot only")


class ShotBreakdown(BaseModel):
    shots: list[ShotCandidate] = Field(..., min_length=1, max_length=settings.max_video_shots_per_scene)


def _format_scene_for_llm(scene: ComposedScenePayload) -> str:
    lines: list[str] = [f"Passage text:\n{scene.selected_text}", ""]

    if scene.location is not None:
        location_line = f"Location: {scene.location.name} -- {scene.location.visual_description}"
        if scene.location.lighting_state:
            location_line += f" Lighting: {scene.location.lighting_state}."
        lines.append(location_line)

    for character in scene.characters:
        bit = f"Character {character.name}: {character.visual_description}"
        if character.emotional_state:
            bit += f" Currently: {character.emotional_state}."
        lines.append(bit)

    if scene.dialogue_script:
        speaking = ", ".join(sorted({line.character_name for line in scene.dialogue_script}))
        lines.append(f"Characters speaking on screen during this span: {speaking}.")

    lines.append(f"Camera framing established by the text: {scene.camera_framing.replace('_', ' ')}")
    lines.append(f"Action across this span: {scene.action_summary}")
    return "\n".join(lines)


def _build_world(scene: ComposedScenePayload) -> VideoWorldPayload:
    """The fixed per-scene identity anchors -- same role as a hand-authored
    WORLD dict. Built once, deterministically, from the already-resolved
    Tier 1/2 state; never touched by the LLM."""
    return VideoWorldPayload(
        characters={c.name: c.visual_description for c in scene.characters},
        location=scene.location.visual_description if scene.location else None,
        look=settings.video_style_suffix,
    )


def _build_prompt(*, camera: str, action: str, light: str, world: VideoWorldPayload) -> str:
    """Assembles one shot's final prompt from the fixed world anchors plus
    this shot's camera/action/light, in a stable field order (identity
    early, style last) so every shot in a scene reads as the same
    continuous take rather than an independently regenerated description."""
    parts: list[str] = [camera, "Part of one continuous cinematic sequence.", action]
    parts.extend(
        f"This is the same individual {name} in every shot -- {description}."
        for name, description in world.characters.items()
    )
    if world.location:
        parts.append(f"The setting stays the same throughout -- {world.location}.")
    parts.append(light)
    parts.append(world.look)
    return " ".join(part.strip() for part in parts if part.strip())


async def generate_video_plan(pool: GpuWorkerPool, scene: ComposedScenePayload) -> VideoPlanPayload:
    """Plans the camera/action/light for each shot via the LLM, then
    deterministically builds `world`, splices it (plus the shared style and
    negative prompt) into every shot's `prompt` -- so appearance and look
    never drift between shots or between scenes."""
    result = await pool.extract_structured(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=_format_scene_for_llm(scene),
        response_schema=ShotBreakdown,
        max_tokens=1024,
        temperature=0.3,
    )

    world = _build_world(scene)
    shots = [
        VideoShotPayload(
            shot_id=shot.shot_id,
            camera=shot.camera,
            action=shot.action,
            light=shot.light,
            prompt=_build_prompt(camera=shot.camera, action=shot.action, light=shot.light, world=world),
        )
        for shot in result.shots
    ]
    return VideoPlanPayload(world=world, shots=shots, negative_prompt=settings.video_negative_prompt)
