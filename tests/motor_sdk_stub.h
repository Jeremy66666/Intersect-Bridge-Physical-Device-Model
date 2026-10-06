#pragma once
// Hardware-free substitutes used ONLY when MOTOR_PATH_TESTING is defined.
#include <map>
#include <vector>
#include <string>
namespace StubMotor {
inline std::map<std::string,int> positions, homes;
inline std::map<std::string,WORD> messages;
inline std::vector<std::pair<std::string,int>> moves;
inline int maximum = 867500;
inline void reset() {
    positions = {{"27266528",69400},{"27266445",72870},{"27266446",131860}};
    homes.clear(); messages.clear(); moves.clear(); maximum = 867500;
}
}
inline int TLI_BuildDeviceList() { return 0; }
inline int CC_Open(const char*) { return 0; }
inline void CC_Close(const char*) {}
inline bool CC_StartPolling(const char*, int) { return true; }
inline void CC_StopPolling(const char*) {}
inline int CC_RequestSettings(const char*) { return 0; }
inline int CC_GetStageAxisMinPos(const char*) { return 0; }
inline int CC_GetStageAxisMaxPos(const char*) { return StubMotor::maximum; }
inline int CC_GetPosition(const char* s) { return StubMotor::positions[s]; }
inline void CC_ClearMessageQueue(const char* s) { StubMotor::messages.erase(s); }
inline int CC_MessageQueueSize(const char* s) { return static_cast<int>(StubMotor::messages.count(s)); }
inline bool CC_GetNextMessage(const char* s, WORD* type, WORD* id, DWORD* data) {
    *type = 2; *id = StubMotor::messages[s]; *data = 0; StubMotor::messages.erase(s); return true;
}
inline int CC_Home(const char* s) {
    ++StubMotor::homes[s]; StubMotor::positions[s] = 0; StubMotor::messages[s] = 0; return 0;
}
inline int CC_MoveToPosition(const char* s, int p) {
    StubMotor::moves.push_back({s,p}); StubMotor::positions[s] = p; StubMotor::messages[s] = 1; return 0;
}
inline int CC_MoveRelative(const char* s, int delta) { return CC_MoveToPosition(s, CC_GetPosition(s)+delta); }
inline int CC_StopImmediate(const char*) { return 0; }
inline int CC_SetBacklash(const char*, int) { return 0; }
