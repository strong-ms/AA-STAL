"""Frame-level Ground evaluation with explicitly optimistic action-based identity matching.

No external dependencies. Input ZIP is read without extracting or executing content.
"""
import argparse
import csv
import itertools
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ACTIONS = ['cut', 'drop', 'grasp', 'handover', 'hold', 'idle', 'place down',
           'pour', 'push', 'shake', 'squash', 'twist']
NAME = re.compile(r'(P\d+_T\d+_V\d+)_C(\d+)_scene(\d+)_part(\d+)_(person_\d+)_actions.json')
WINDOW = re.compile(r'frames_(\d+)_to_(\d+)')
SUBSETS = [('', set()), ('_NO_DROP', {'drop'}), ('_NO_IDLE', {'idle'}),
           ('_NO_IDLE_DROP', {'idle', 'drop'}),
           ('_NO_IDLE_HOLD', {'idle', 'hold'}),
           ('_NO_IDLE_HOLD_DROP', {'idle', 'hold', 'drop'})]


def write_csv(path, rows, fields=None):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def class_metrics(matrix):
    rows = []
    for label in ACTIONS:
        tp = matrix[label, label]
        support = sum(n for (a, b), n in matrix.items() if a == label)
        predicted = sum(n for (a, b), n in matrix.items() if b == label)
        covered = support - matrix[label, 'no_prediction']
        precision = tp / predicted if predicted else 0
        recall = tp / support if support else 0
        rows.append(dict(action=label, support_frames=support, predicted_frames=predicted,
                         tp=tp, fp=predicted - tp, fn=support - tp,
                         covered_frames=covered, coverage_ratio=covered / support if support else 0,
                         precision=precision, recall=recall,
                         f1=2 * tp / (predicted + support) if predicted + support else 0))
    return rows


def averages(rows):
    result = []
    for suffix, excluded in SUBSETS:
        selected = [r for r in rows if r['action'] not in excluded]
        support = sum(r['support_frames'] for r in selected)
        tp, fp, fn = (sum(r[k] for r in selected) for k in ('tp', 'fp', 'fn'))
        coverage = sum(r['covered_frames'] for r in selected) / support if support else 0
        for mode in ('MACRO', 'MICRO', 'WEIGHTED'):
            if mode == 'MICRO':
                values = dict(precision=tp / (tp + fp) if tp + fp else 0,
                              recall=tp / (tp + fn) if tp + fn else 0,
                              f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
            else:
                weights = [1 if mode == 'MACRO' else r['support_frames'] for r in selected]
                denominator = sum(weights)
                values = {k: sum(r[k] * w for r, w in zip(selected, weights)) / denominator
                          if denominator else 0 for k in ('precision', 'recall', 'f1')}
            result.append(dict(average=mode + '_AVG' + suffix, support_frames=support,
                               tp=tp if mode == 'MICRO' else '',
                               fp=fp if mode == 'MICRO' else '',
                               fn=fn if mode == 'MICRO' else '', coverage_ratio=coverage, **values))
    return result


def collapse_trace(labels, excluded):
    """Filter the selected vocabulary, then merge equal consecutive labels."""
    return [label for label, _ in itertools.groupby(x for x in labels if x not in excluded)]


def sequence_scores(reference, prediction):
    """Unit-cost Levenshtein and order-preserving longest common subsequence."""
    edit = list(range(len(prediction) + 1))
    lcs = [0] * (len(prediction) + 1)
    for i, expected in enumerate(reference, 1):
        next_edit, next_lcs = [i], [0]
        for j, actual in enumerate(prediction, 1):
            next_edit.append(min(edit[j] + 1, next_edit[-1] + 1,
                                 edit[j - 1] + (expected != actual)))
            next_lcs.append(lcs[j - 1] + 1 if expected == actual else max(lcs[j], next_lcs[-1]))
        edit, lcs = next_edit, next_lcs
    distance, matched = edit[-1], lcs[-1]
    longest = max(len(reference), len(prediction))
    return dict(gt_events=len(reference), predicted_events=len(prediction),
                exact_match=int(reference == prediction), edit_distance=distance,
                edit_similarity=1 - distance / longest if longest else 1,
                lcs_matches=matched,
                lcs_precision=matched / len(prediction) if prediction else int(not reference),
                lcs_recall=matched / len(reference) if reference else int(not prediction),
                lcs_f1=2 * matched / (len(reference) + len(prediction)) if longest else 1)


def trace_evaluation(gt, predictions):
    traces, averages = [], []
    for suffix, excluded in SUBSETS:
        subset = 'ALL' + suffix
        for video, people in sorted(gt.items()):
            for person, frames in sorted(people.items()):
                ordered = sorted(frames)
                if ordered != list(range(ordered[0], ordered[-1] + 1)):
                    raise ValueError(f'Trace has GT gaps: {video}/{person}; explicit gap alignment required')
                reference = collapse_trace((frames[f] for f in ordered), excluded)
                prediction = collapse_trace((predictions[video, person].get(f, 'no_prediction')
                                              for f in ordered), excluded)
                traces.append(dict(subset=subset, video=video, person=person,
                                   **sequence_scores(reference, prediction),
                                   gt_sequence=json.dumps(reference), predicted_sequence=json.dumps(prediction)))
        rows = [r for r in traces if r['subset'] == subset]
        averages.append(dict(subset=subset, traces=len(rows),
                             exact_trace_accuracy=sum(r['exact_match'] for r in rows) / len(rows),
                             mean_edit_distance=sum(r['edit_distance'] for r in rows) / len(rows),
                             mean_edit_similarity=sum(r['edit_similarity'] for r in rows) / len(rows),
                             mean_lcs_precision=sum(r['lcs_precision'] for r in rows) / len(rows),
                             mean_lcs_recall=sum(r['lcs_recall'] for r in rows) / len(rows),
                             mean_lcs_f1=sum(r['lcs_f1'] for r in rows) / len(rows)))
    return traces, averages


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--ground-truth', type=Path,
                        default=Path(globals().get('__file__', 'evaluate_actions.py')).with_name('ground_truth_actions.csv'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    gt = defaultdict(lambda: defaultdict(dict))
    with args.ground_truth.open() as stream:
        for row in csv.DictReader(stream):
            label = row['verb'].lower().replace('_', ' ')
            if label not in ACTIONS:
                raise ValueError(f'Unsupported GT verb: {label}')
            frames = gt[Path(row['source_file']).stem][row['person']]
            for frame in range(int(row['start']), int(row['end']) + 1):
                if frame in frames:
                    raise ValueError('Overlapping GT intervals')
                frames[frame] = label

    clips = defaultdict(dict)
    with zipfile.ZipFile(args.predictions) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or '__MACOSX' in entry.filename.split('/'):
                continue
            name = Path(entry.filename).name
            if not name.endswith('.json'):
                continue
            match = NAME.fullmatch(name)
            if not match:
                raise ValueError(f'Unrecognized filename: {name}')
            video, camera, scene, part, track = match.groups()
            if camera != '4' or scene != '1' or video not in gt:
                raise ValueError(f'Cannot align video automatically: {name}')
            key = (video, int(part))
            if track in clips[key]:
                raise ValueError(f'Duplicate prediction file: {name}')
            windows = []
            for key_window, value in json.loads(archive.read(entry)).items():
                window = WINDOW.fullmatch(key_window)
                if not window:
                    raise ValueError(f'Invalid window: {key_window}')
                start, end = map(int, window.groups())
                labels = value['actions']
                if not 0 <= start < end <= 300 or len(labels) != 1:
                    raise ValueError(f'Invalid prediction: {name}/{key_window}')
                if labels[0] not in ACTIONS + ['unknown']:
                    raise ValueError(f'Invalid label: {labels[0]}')
                windows.append((start, end, labels[0]))
            # Overlap resolution: nearest window center, earlier start wins ties.
            frames = {}
            for frame in range(300):
                candidates = [w for w in windows if w[0] <= frame < w[1]]
                if candidates:
                    best = min(candidates, key=lambda w: (abs((w[0] + w[1] - 1) / 2 - frame), w[0]))
                    frames[frame] = best[2]
            clips[key][track] = frames

    matrix = Counter()
    predicted_traces = defaultdict(dict)
    mappings, alternatives, video_rows = [], [], []
    for video, people in sorted(gt.items()):
        video_matrix = Counter()
        final = max(max(frames) for frames in people.values())
        for part in range(1, final // 300 + 2):
            offset = (part - 1) * 300
            active = sorted(p for p, frames in people.items() if any(offset <= f < offset + 300 for f in frames))
            tracks = clips.get((video, part), {})
            track_ids = sorted(tracks)
            # Assign injectively; missing tracks are explicit, surplus tracks reported.
            candidates = track_ids + [f'missing_{i}' for i in range(max(0, len(active) - len(track_ids)))]
            scores = []
            for assignment in itertools.permutations(candidates, len(active)):
                score = sum(
                    tracks.get(t, {}).get(f - offset, 'no_prediction') == label
                    for p, t in zip(active, assignment)
                    for f, label in people[p].items() if offset <= f < offset + 300)
                scores.append((score, assignment))
            scores.sort(key=lambda item: (-item[0], item[1]))
            best_score, best = scores[0]
            support = sum(offset <= f < offset + 300 for p in active for f in people[p])
            gap = best_score - scores[1][0] if len(scores) > 1 else None
            for rank, (score, assignment) in enumerate(scores, 1):
                alternatives.append(dict(video=video, part=part, rank=rank,
                                         mapping=json.dumps(dict(zip(active, assignment))),
                                         correct_frames=score, support_frames=support))
            for person, track in zip(active, best):
                mappings.append(dict(video=video, part=part, gt_person=person,
                                     track=track, best_correct_frames=best_score,
                                     next_best_gap_frames=gap, tied=sum(s[0] == best_score for s in scores),
                                     unused_tracks=';'.join(t for t in track_ids if t not in best)))
                for frame, label in people[person].items():
                    if offset <= frame < offset + 300:
                        pred = tracks.get(track, {}).get(frame - offset, 'no_prediction')
                        predicted_traces[video, person][frame] = pred
                        video_matrix[label, pred] += 1
        matrix.update(video_matrix)
        support = sum(video_matrix.values())
        missing = sum(n for (a, b), n in video_matrix.items() if b == 'no_prediction')
        video_classes = class_metrics(video_matrix)
        video_rows.append(dict(video=video, support_frames=support,
                               accuracy=sum(n for (a, b), n in video_matrix.items() if a == b) / support,
                               coverage_ratio=1 - missing / support,
                               macro_f1=sum(r['f1'] for r in video_classes) / len(ACTIONS),
                               no_prediction_frames=missing))

    per_class = class_metrics(matrix)
    aggregate_rows = averages(per_class)
    trace_rows, trace_averages = trace_evaluation(gt, predicted_traces)
    total = sum(matrix.values())
    summary = dict(protocol='optimistic action-based matching per clip; identities not independently verified',
                   alignment='scene1 starts at 0; part offsets 300 frames; GT End inclusive; predictions End exclusive',
                   accuracy=sum(n for (a, b), n in matrix.items() if a == b) / total,
                   macro_f1=sum(r['f1'] for r in per_class) / len(per_class),
                   weighted_f1=sum(r['f1'] * r['support_frames'] for r in per_class) / total,
                   support_person_frames=total,
                   unknown_rate=sum(n for (a, b), n in matrix.items() if b == 'unknown') / total,
                   no_prediction_rate=sum(n for (a, b), n in matrix.items() if b == 'no_prediction') / total,
                   prediction_files=sum(len(v) for v in clips.values()))
    summary.update(videos_evaluated=len(video_rows),
                   accuracy_macro_per_video=sum(r['accuracy'] for r in video_rows) / len(video_rows),
                   coverage_macro_per_video=sum(r['coverage_ratio'] for r in video_rows) / len(video_rows),
                   averages=aggregate_rows, trace_averages=trace_averages)
    write_csv(args.output / 'person_mapping.csv', mappings)
    write_csv(args.output / 'mapping_alternatives.csv', alternatives)
    write_csv(args.output / 'per_class.csv', per_class)
    write_csv(args.output / 'averages.csv', aggregate_rows)
    write_csv(args.output / 'per_video.csv', video_rows)
    write_csv(args.output / 'per_trace.csv', trace_rows)
    write_csv(args.output / 'trace_averages.csv', trace_averages)
    write_csv(args.output / 'confusion_matrix.csv', [dict(gt_action=a, **{b: matrix[a, b] for b in ACTIONS + ['unknown', 'no_prediction']}) for a in ACTIONS])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = [
        'GROUND ACTION RECOGNITION — FRAME LEVEL',
        'OPTIMISTIC: person identities inferred from agreement with GT actions.',
        'Alignment assumed: 30 FPS, scene1 offset 0, 300 frames per part, GT End inclusive.',
        f'Videos evaluated: {len(video_rows)} | Prediction files: {summary["prediction_files"]}',
        f'Annotated person-frames: {total}',
        f'Accuracy pooled: {summary["accuracy"]:.4f}',
        f'Accuracy macro per video: {summary["accuracy_macro_per_video"]:.4f}',
        f'Coverage macro per video: {summary["coverage_macro_per_video"]:.4f}',
        f'Unknown rate: {summary["unknown_rate"]:.4f}',
        'Subset accuracy / sample Jaccard: same as accuracy for singleton labels; abstentions treated as empty sets.',
        'No L0/L1 hierarchy is defined for Ground.', '',
        f'{"action":<24} {"precision":>9} {"recall":>8} {"f1":>8} {"support":>8} {"tp":>7} {"fp":>7} {"fn":>7} {"coverage":>9}',
    ]
    for row in per_class + [dict(r, action=r['average']) for r in aggregate_rows]:
        lines.append(f'{row["action"]:<24} {row["precision"]:9.4f} {row["recall"]:8.4f} '
                     f'{row["f1"]:8.4f} {row["support_frames"]:8d} {str(row["tp"]):>7} '
                     f'{str(row["fp"]):>7} {str(row["fn"]):>7} {row["coverage_ratio"]:9.4f}')
    lines.extend(['', 'Excluded-class averages retain errors across excluded/included classes as FP/FN.',
                  'Unknown and no_prediction cause FN, not FP for real action classes; unknown counts as covered.',
                  'Macro averages give equal weight to classes (including zero scores); weighted averages use GT support.',
                  'No GT videos are skipped; missing predictions are scored as no_prediction.',
                  'These are frame classification scores, not temporal event detection metrics.'])
    lines.extend(['', 'FULL PERSON TRACES — ORDER ONLY, CONSECUTIVE DUPLICATES COLLAPSED',
                  'One trace per GT person per original video; clips joined using the same inferred mapping.',
                  'No total ordering is imposed between simultaneous actions of different people.',
                  'Unknown/no_prediction retained as error events; no smoothing or minimum duration filter.',
                  'Sequence subsets filter labels on BOTH sides before collapsing: a different protocol from frame averages.',
                  'Levenshtein uses unit insertion/deletion/substitution costs; similarity = 1 - distance / max(lengths).',
                  'LCS matches actions in order, without requiring adjacency. Averages give equal weight to traces.',
                  'Empty/empty sequences score 1; empty/nonempty scores 0 for normalized similarities.',
                  f'{"subset":<24} {"traces":>6} {"exact":>8} {"edit_sim":>9} {"LCS_P":>8} {"LCS_R":>8} {"LCS_F1":>8}'])
    for row in trace_averages:
        lines.append(f'{row["subset"]:<24} {row["traces"]:6d} {row["exact_trace_accuracy"]:8.4f} '
                     f'{row["mean_edit_similarity"]:9.4f} {row["mean_lcs_precision"]:8.4f} '
                     f'{row["mean_lcs_recall"]:8.4f} {row["mean_lcs_f1"]:8.4f}')
    (args.output / 'evaluation_results.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
