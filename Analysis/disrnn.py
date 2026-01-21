# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 12:40:00 2026

@author: svc_ccg
"""

import copy
from absl import app
from absl import flags
from disentangled_rnns.library import disrnn
from disentangled_rnns.library import plotting
from disentangled_rnns.library import rnn_utils
import optax
from DynamicRoutingAnalysisUtils import getIsStandardRegimen,getSessionData
from RNNmodelHPC import getRNNSessions


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"


summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))
drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
isStandardRegimen = getIsStandardRegimen(summaryDf)
mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass'] ]['mouse id'])

maxTrainSessions = 20
mouseIds = []
for mouseId in mice:
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    sessions = getRNNSessions(mouseId,df)
    if len(sessions) > maxTrainSessions:
        mouseIds.append(mouseId)







FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "n_steps_per_session", 200, "Number of steps per session in the dataset."
)
flags.DEFINE_integer("n_sessions", 300, "Number of sessions in the dataset.")
flags.DEFINE_float("learning_rate", 1e-3, "Optimizer learning rate.")
flags.DEFINE_integer("n_warmup_steps", 1000, "Number of training warmup steps.")
flags.DEFINE_integer(
    "n_training_steps", 3000, "Number of main training steps."
)







xs: np.typing.NDArray[np.number],
ys: np.typing.NDArray[np.number],
y_type: Literal['categorical', 'scalar', 'mixed'] = 'categorical',
n_classes: Optional[int] = None,
x_names: Optional[list[str]] = None,
y_names: Optional[list[str]] = None,
batch_size: Optional[int] = None,
batch_mode: Literal['single', 'rolling', 'random'] = 'single',

rnn_utils.DatasetRNN
































