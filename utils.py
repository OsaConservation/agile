import pandas as pd
from perch_hoplite.db.datatypes import LabelType, Annotation

def insert_annotations_from_file(db, filepath, handle_duplicates = "error"):
    df = pd.read_csv("./data/audio/annotations.csv")
    df["filename"] = df["audiofilename"].str.split("/", n=1).str[-1]
    files = df["filename"].to_list()
    ids = []
    for file in files:
        ids.append(get_recording_id(file))
    dfsubset = df[["start", "end"]]

    df["recording_id"] = df["filename"].apply(get_recording_id)
    df["offsets"] = dfsubset.apply(list,axis=1)
    df["label_type"] = LabelType.POSITIVE


    df = df.rename(columns = {'label:species' : 'label'} )
    df["provenance"] = "silencio"

    df = df[["recording_id", "offsets", "label", "label_type", "provenance"]]


    print('Annotations before saving :', len(db.get_all_annotations()))

    for row in df.itertuples():
        db.insert_annotation(row.recording_id, row.offsets, row.label, row.label_type, row.provenance)

    print('Annotations after saving  :', len(db.get_all_annotations()))


def get_recording_id(filename):
    recordings = db.get_all_recordings()
    result = list(filter(lambda x: x.filename == filename, recordings))
    if result:
        result = result[0]
        return result.id
    else:
        return None
