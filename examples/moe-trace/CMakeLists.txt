set(TARGET llama-moe-trace)
set(MOEVM_PHASE0_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../moevm/phase0" CACHE PATH
    "Directory of the MoEVM repository's phase0/ sources (moevm_runtime.cpp/.hpp)")
add_executable(${TARGET} moe-trace.cpp)
target_sources(${TARGET} PRIVATE ${MOEVM_PHASE0_DIR}/moevm_runtime.cpp)
target_include_directories(${TARGET} PRIVATE ${MOEVM_PHASE0_DIR})
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
