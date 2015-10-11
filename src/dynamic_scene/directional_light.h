#ifndef CMU462_DYNAMICSCENE_DIRECTIONALLIGHT_H
#define CMU462_DYNAMICSCENE_DIRECTIONALLIGHT_H

#include "scene.h"

namespace CMU462 { namespace DynamicScene {

class DirectionalLight : public SceneLight {
 public:
  DirectionalLight(const Color& color, const Matrix4x4& transform) {
    this->color = color;
    this->transform = transform;
  }

  void opengl_init_light(GLenum lightIndex) const {
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrixd(&transform(0, 0));

    glLightfv(lightIndex, GL_DIFFUSE, &color.r);
    const GLfloat arg2[] = {1.0, 1.0, 1.0, 0.0};
    glLightfv(lightIndex, GL_POSITION, arg2);
    const GLfloat arg3[] = {0.0, 0.0, -1.0}; // default direction
    glLightfv(lightIndex, GL_SPOT_DIRECTION, arg3);
    glEnable(lightIndex);

    glPopMatrix();
  }

  StaticScene::SceneLight *get_static_light() const {
    return nullptr;
  }

 private:
  Color color;
  Matrix4x4 transform; // local to world
};

} // namespace DynamicScene
} // namespace CMU462

#endif //CMU462_DYNAMICSCENE_DIRECTIONALLIGHT_H
