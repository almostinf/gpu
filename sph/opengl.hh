#pragma once
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#endif
#include "sph.hh"

inline void draw_particles(const std::vector<Particle>& particles){
    glClearColor(0.9f,0.9f,0.9f,1);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);glLoadIdentity();
    glOrtho(0,domain_width,0,domain_height,-1,1);
    glMatrixMode(GL_MODELVIEW);glLoadIdentity();
    glEnable(GL_POINT_SMOOTH);glPointSize(kernel_radius);
    glColor3f(0.2f,0.6f,1.0f);glBegin(GL_POINTS);
    for(const auto& p:particles)glVertex2f(p.position[0],p.position[1]);
    glEnd();glutSwapBuffers();
}
