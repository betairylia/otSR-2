# Osu API
from ossapi import *
import os
import time
import pickle
import sys

api = Ossapi(int(os.environ['OSUAPIv2_APP_ID']), os.environ['OSUAPIv2_APP_SECRET'], os.environ['OSUAPIv2_APP_CALLBACK'], token_directory = "token/")
global_wait_time = 6

def download_rep(replayID, name = None):

    if name is None:
        name = replayID

    dest = f"replays/{name}.osr"

    if os.path.exists(dest):
        return False

#     print(api.BASE_URL)
#     rep = api.session.get(f"{api.BASE_URL}/scores/taiko/{replayID}/download")
    
    try:
        rep = api.download_score(mode = "taiko", score_id = replayID, raw = True)
        with open(dest, "wb") as f:
            f.write(rep)
    except ValueError as err:
#         print(err)
        open("download.log","a").write("-----%s-----\n%s\n" % (name, str(err)))
        if not "None" in str(err):
            time.sleep(global_wait_time)
        return False
            
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

# top5K = GetRanking(5000)
# CN400 = GetRanking(400, country = "CN")
# uniqueIDs = list(set([u.id for u in top5K] + [u.id for u in CN400]))

# pickle.dump(uniqueIDs, open("ids.pkl", "wb"))
uniqueIDs = pickle.load(open("ids.pkl", "rb"))
print(uniqueIDs)

metadict = {}
# if os.path.exists("meta.pkl"):
#     metadict = pickle.load(open("meta.pkl", "rb"))

def GrabFromBP(uid, amount):
    print("Grab BP maps from uid %d (%s): " % (uid, api.user(uid).username), end="")
    best = api.user_scores(uid, "best", mode = "taiko", limit = 20)[:amount]
    for b in best:
        scoreid = b.id
        metadict[b.id] = b
        
#         if download_rep(scoreid, f"{b.user().username} =+= {b.beatmapset.artist} - {b.beatmapset.title} [{b.beatmap.version}] =+= {b.id}"):
        if download_rep(scoreid, f"{b.user().username} - {b.id}"):
            print("-", end="")
            sys.stdout.flush()
            time.sleep(global_wait_time)
        else:
            print("X", end="")
            sys.stdout.flush()

#         if download_rep(scoreid, f"{b.user().username} - {b.id}"):
            # Wait if we accessed the download page
#             time.sleep(5)
#     pickle.dump(metadict, open("meta.pkl", "wb"))

start_from = int(open("start.uid", "r").readline())
started = False

for i, uid in enumerate(uniqueIDs):
    open("start.uid", "w").write("%d" % uid)
    if uid == start_from:
        started = True
    if started:
        print("\n[%4d / %4d] " % (i, len(uniqueIDs)), end = "")
        GrabFromBP(uid, 8)
