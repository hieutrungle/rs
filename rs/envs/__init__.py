from rs.envs.conference_2ue_allocation import Conference2UEAllocation
from rs.envs.conference_4ue_allocation import Conference4UEAllocation
from rs.envs.classroom_2ue import Classroom2UE
from rs.envs.classroom_4ue import Classroom4UE
from rs.envs.classroom import Classroom
from rs.envs.classroom_eval import ClassroomEval
from rs.envs.data_center import TwoAgentDataCenter

ENV_IDS = {
    "conference_2ue": Conference2UEAllocation,
    "conference_4ue": Conference4UEAllocation,
    "classroom_2ue": Classroom2UE,
    "classroom_4ue": Classroom4UE,
    "classroom": Classroom,
    "classroom_eval": ClassroomEval,
    "data_center": TwoAgentDataCenter,
}
