import sqlite3

def init_labels(db_path):
    # initialize Database Parameters
    Description = 'SBERT, HDBSCAN, Silhouette, With Replacement'
    Comment = 'min_cluster_size=11, n_comp=17, n_neighbors:22'

    db_path = 'data/'
    filepath = db_path + 'ConstructMapping.db'
    conn = sqlite3.connect(filepath)
    cursor = conn.cursor()

    cursor.execute("INSERT INTO Labelset (Description, Comment) VALUES (?,?)", (Description, Comment))
    conn.commit()

    labelset_id = cursor.lastrowid

    cursor.execute("INSERT INTO Label (ResponseID, LabelSetID) SELECT ResponseID, ? FROM Response WHERE IsValid", (labelset_id,))

    rows = cursor.fetchall()

    conn.commit()
    cursor.close()
    conn.close()
    return labelset_id