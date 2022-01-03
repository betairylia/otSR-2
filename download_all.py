# Osu API
from ossapi import *
import os
import time
import pickle

api = OssapiV2(int(os.environ['OSUAPIv2_APP_ID']), os.environ['OSUAPIv2_APP_SECRET'], os.environ['OSUAPIv2_APP_CALLBACK'])

def download_rep(replayID, name = None):

    if name is None:
        name = replayID

    dest = f"replays/{name}.osr"

    if os.path.exists(dest):
        return False

    rep = api.session.get(f"{api.BASE_URL}/scores/taiko/{replayID}/download")
    with open(dest, "wb") as f:
        f.write(rep.content)

    return True

def GetRanking(count = 100, mode = "taiko", country = None):
    
    cnt = 0
    cursor = None
    result = []
    
    while cnt < count:
        r = api.ranking(mode, RankingType.PERFORMANCE, country = country, cursor = cursor)
        cursor = r.cursor
        cnt += len(r.ranking)
        for entry in r.ranking:
            result.append(entry.user)
    
    return result

top5K = GetRanking(5000)
CN400 = GetRanking(400, country = "CN")
uniqueIDs = list(set([u.id for u in top5K] + [u.id for u in CN400]))

metadict = {}
if os.path.exists("meta.pkl"):
    metadict = pickle.load(open("meta.pkl", "rb"))

def GrabFromBP(uid, amount):
    print("Grab BP maps from uid %d (%s)" % (uid, api.user(uid).username))
    best = api.user_scores(uid, "best", mode = "taiko")[:amount]
    for b in best:
        scoreid = b.id
        metadict[b.id] = b

        if download_rep(scoreid, f"{b.user().username} - {b.id}"):
            # Wait if we accessed the download page
            time.sleep(15)
    pickle.dump(metadict, open("meta.pkl", "wb"))

for uid in uniqueIDs:
    GrabFromBP(uid, 15)
