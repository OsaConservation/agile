import pandas as pd
from perch_hoplite.db.datatypes import LabelType, Annotation

def insert_annotations_from_file(db, filepath, handle_duplicates = "error"):
    df = pd.read_csv(filepath)
    df["filename"] = df["audiofilename"].str.split("/", n=1).str[-1]

    dfsubset = df[["start", "end"]]

    df["recording_id"] = df["filename"].apply(get_recording_id)
    df["offsets"] = dfsubset.apply(list,axis=1)
    df["label_type"] = LabelType.POSITIVE


    df = df.rename(columns = {'label:species' : 'label'} )
    df["provenance"] = "silencio"

    df = df[["recording_id", "offsets", "label", "label_type", "provenance"]]


    print('Annotations before saving :', len(db.get_all_annotations()))

    for row in df.itertuples():
        db.insert_annotation(row.recording_id, row.offsets, row.label, row.label_type, row.provenance, handle_duplicates=handle_duplicates)

    print('Annotations after saving  :', len(db.get_all_annotations()))


def get_recording_id(filename, db):
    recordings = db.get_all_recordings()
    result = list(filter(lambda x: x.filename == filename, recordings))
    if result:
        result = result[0]
        return result.id
    else:
        return None

def filter_results(project, results, db):

    windows = [int(r.window_id) for r in results.search_results]
    embeds = [db.get_window(w) for w in windows]
    records = [db.get_recording(w.recording_id) for w in embeds]
    deploys = [db.get_deployment(w.deployment_id) for w in records]

    keep = []

    for deploy in deploys:
        if deploy.project == project:
            keep.append(True)
        else:
            keep.append(False)

    filtered = [w for w, m in zip(results, keep) if m]
    print(keep)

    return filtered
