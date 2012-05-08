
set(ffusion_SOURCES main.cpp OrbitCamera.cpp DemoBase.cpp FreenectFusion.cpp)

cuda_compile(CUDA_FILES FreenectFusion_kernel.cu)

include_directories(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR} ${Freenect_INCLUDE_DIR})
add_executable(ffusion ${ffusion_SOURCES} ${CUDA_FILES})
target_link_libraries(ffusion ${OPENGL_LIBRARIES}
                              ${GLUT_LIBRARIES}
                              ${CUDA_LIBRARIES}
                              ${FreenectSync_LIBRARY})
