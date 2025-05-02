#%%
import os
import numpy as np
import pandas as pd
import npc_lims
import npc_sessions


# %%
sessionInfo = npc_lims.get_session_info()

#%%
mouse = '728917'
#mouse = '762124','741611','748098'
sessions = [s for s in sessionInfo if s.subject==mouse and s.is_ephys]
sessionIds = [s.id for s in sessions if s.is_annotated and len(s.issues)==0]

# %%
session = npc_sessions.DynamicRoutingSession(sessions[0].id)
 
#%%
trialsDf = session.trials[:]

#%%
unitsDf = session.units[:]

# %%
