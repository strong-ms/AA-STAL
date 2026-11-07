"""Export a wide object-centric CSV with clip-scoped object identities.

Relative timestamps; association evidence is not proof of manipulation.
"""
import csv
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path


def relative_timestamp(frame, fps=30):
    milliseconds = round(frame * 1000 / fps)
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}'


def export_object_event_log(predictions_zip, associations_path, output_dir, fps=30):
    """Read model predictions plus association JSONs; never use GT person matching."""
    associations_path, output_dir = Path(associations_path), Path(output_dir)
    if fps != 30:
        raise ValueError('This Ground exporter assumes 300-frame parts at 30 FPS.')
    association_data = {}

    def add_association(name, data):
        if name in association_data:
            raise ValueError(f'Duplicate association file: {name}')
        association_data[name] = data

    if associations_path.is_dir():
        for path in associations_path.rglob('*_associations.json'):
            add_association(path.name, json.loads(path.read_text()))
    elif associations_path.is_file() and zipfile.is_zipfile(associations_path):
        with zipfile.ZipFile(associations_path) as archive:
            for entry in archive.infolist():
                if '__MACOSX' not in entry.filename.split('/') and entry.filename.endswith('_associations.json'):
                    add_association(Path(entry.filename).name, json.loads(archive.read(entry)))
    else:
        raise FileNotFoundError(f'Association folder or ZIP not found: {associations_path}')

    events, objects, object_types = [], {}, set()
    pattern = re.compile(r'((P\d+_T\d+_V\d+)_C4_scene1_part(\d+))_(person_\d+)_actions.json')
    with zipfile.ZipFile(predictions_zip) as archive:
        inputs = []
        for entry in archive.infolist():
            if entry.is_dir() or '__MACOSX' in entry.filename.split('/') or not entry.filename.endswith('.json'):
                continue
            match = pattern.fullmatch(Path(entry.filename).name)
            if not match:
                raise ValueError(f'Unsupported prediction filename: {entry.filename}')
            clip, video, part, person = match.groups()
            association_name = clip + '_associations.json'
            if association_name not in association_data:
                raise FileNotFoundError(f'Missing association file: {association_name}; no partial log exported')
            inputs.append((clip, video, int(part), person, json.loads(archive.read(entry))))
    seen = set()
    for clip, video, part, person, predictions in sorted(inputs):
        if (clip, person) in seen:
            raise ValueError(f'Duplicate clip/person: {clip}/{person}')
        seen.add((clip, person))
        data = association_data[clip + '_associations.json']
        if 'frame_associations' not in data or 'window_associations' not in data:
            raise ValueError(f'Incomplete association schema: {clip}')
        accepted = defaultdict(list)
        for item in data['window_associations']:
            if item.get('accepted') and item['person_id'] == person:
                accepted[item['object_id']].append((item['start_frame'], item['end_frame']))
        windows = []
        for key, value in predictions.items():
            match = re.fullmatch(r'frames_(\d+)_to_(\d+)', key)
            if not match or len(value.get('actions', [])) != 1:
                raise ValueError(f'Invalid prediction: {clip}/{person}/{key}')
            start, end = map(int, match.groups())
            if not 0 <= start < end <= 300:
                raise ValueError(f'Invalid interval: {key}')
            windows.append((start, end, value['actions'][0]))
        if not windows:
            continue
        person_oid = f'{clip}::{person}'
        objects[person_oid] = dict(object_id=person_oid, object_type='person', clip_id=clip, local_track_id=person)
        states = []
        # Missing leading/intermediate predictions are explicit; trailing extent is unknown.
        for frame in range(max(w[1] for w in windows)):
            candidates = [w for w in windows if w[0] <= frame < w[1]]
            action = min(candidates, key=lambda w: (abs((w[0] + w[1] - 1) / 2 - frame), w[0]))[2] if candidates else 'no_prediction'
            links = set()
            for item in data['frame_associations'].get(str(frame), []):
                if item['person_id'] != person:
                    continue
                local_id = item['object_id']
                if not any(start <= frame < end for start, end in accepted.get(local_id, [])):
                    continue
                kind = item.get('object_class') or 'unknown'
                if kind == 'person':
                    raise ValueError('Non-person association expected')
                oid = f'{clip}::{local_id}'
                obj = dict(object_id=oid, object_type=kind, clip_id=clip, local_track_id=local_id)
                if oid in objects and objects[oid] != obj:
                    raise ValueError(f'Inconsistent object type: {oid}')
                objects[oid] = obj
                object_types.add(kind)
                links.add((kind, oid))
            states.append((action, tuple(sorted(links))))
        start = 0
        for end in range(1, len(states) + 1):
            if end < len(states) and states[end] == states[start]:
                continue
            action, links = states[start]
            offset = (part - 1) * 300
            row = dict(video_id=video, clip_id=clip, person_id=person,
                       timestamp=relative_timestamp(offset + start, fps),
                       end_timestamp=relative_timestamp(offset + end, fps),
                       start_seconds=(offset + start) / fps, end_seconds=(offset + end) / fps,
                       duration_seconds=(end - start) / fps, action=action,
                       local_start_frame=start, local_end_frame=end,
                       association_status='associated' if links else 'no_accepted_association',
                       **{'object::person': json.dumps([person_oid])})
            for kind, oid in links:
                row.setdefault('object::' + kind, []).append(oid)
            for key, value in list(row.items()):
                if key.startswith('object::') and isinstance(value, list):
                    row[key] = json.dumps(value)
            events.append(row)
            start = end
    if not events:
        raise ValueError('No events to export')
    events.sort(key=lambda r: (r['video_id'], r['start_seconds'], r['clip_id'], r['person_id']))
    relations = []
    for index, row in enumerate(events, 1):
        row['event_id'] = f'event_{index:07d}'
        for key, value in row.items():
            if key.startswith('object::'):
                for oid in json.loads(value):
                    relations.append(dict(event_id=row['event_id'], object_id=oid,
                                          qualifier='actor' if key == 'object::person' else 'associated'))
    output_dir.mkdir(parents=True, exist_ok=True)
    base = ['event_id'] + [k for k in events[0] if k != 'event_id' and not k.startswith('object::')]
    fields = base + ['object::person'] + ['object::' + kind for kind in sorted(object_types)]
    for filename, rows, columns in [
        ('object_centric_event_log.csv', events, fields),
        ('event_log_objects.csv', sorted(objects.values(), key=lambda x: x['object_id']), list(next(iter(objects.values())))),
        ('event_object_relations.csv', relations, ['event_id', 'object_id', 'qualifier']),
    ]:
        with (output_dir / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, restval='[]')
            writer.writeheader()
            writer.writerows(rows)
    metadata = dict(events=len(events), objects=len(objects), clip_persons=len(seen),
                    timestamp_basis='relative to original video; 30 FPS; scene1 offset 0; 300-frame parts',
                    segmentation='split on action OR accepted associated object-set change; no smoothing',
                    identity_scope='clip only; no cross-clip re-identification; no GT-derived identities',
                    relation_semantics='geometric association, not proof of manipulation',
                    format='wide object-centric CSV plus object inventory and relation table; not OCEL 2.0 serialization',
                    object_types=['person'] + sorted(object_types))
    (output_dir / 'event_log_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata
