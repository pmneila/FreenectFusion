
set(ffusion_SOURCES main.cpp OrbitCamera.cpp DemoBase.cpp FreenectFusion.cpp
    MarchingCubes.cpp)

cuda_compile(CUDA_FILES FreenectFusion_kernel.cu MarchingCubes_kernel.cu)

include_directories(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR}
                    ${CUDA_INCLUDE_DIRS} ${Freenect_INCLUDE_DIR})
add_executable(ffusion ${ffusion_SOURCES} ${CUDA_FILES})
target_link_libraries(ffusion ${OPENGL_LIBRARIES}
                              ${GLUT_LIBRARIES}
                              ${CUDA_LIBRARIES}
                              ${FreenectSync_LIBRARY})
