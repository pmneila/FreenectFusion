
#ifndef _GLHEADERS_H
#define _GLHEADERS_H

#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glut.h>
#endif

#endif // _GLHEADERS_H
