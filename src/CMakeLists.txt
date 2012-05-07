
set(ffusion_SOURCES main.cpp OrbitCamera.cpp DemoBase.cpp)

include_directories(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR} ${Boost_INCLUDE_DIR})
add_executable(ffusion ${ffusion_SOURCES})
target_link_libraries(ffusion ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
