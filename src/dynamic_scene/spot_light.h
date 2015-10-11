#ifndef CMU462_DYNAMICSCENE_SPOTLIGHT_H
#define CMU462_DYNAMICSCENE_SPOTLIGHT_H

#include "scene.h"

namespace CMU462 { namespace DynamicScene {

class SpotLight : public SceneLight {
 public:
  SpotLight(const Color& color, const Matrix4x4& transform) {
    this->color = color;
    this->transform = transform;
  }

  void opengl_init_light(GLenum lightIndex) const {
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrixd(&transform(0, 0));

    glLightfv(lightIndex, GL_DIFFUSE, &color.r);
    const GLfloat arg1[] = {0.0, 0.0, 0.0, 1.0};
    glLightfv(lightIndex, GL_POSITION, arg1);
    glEnable(lightIndex);

    glPopMatrix();
  }

  StaticScene::SceneLight *get_static_light() const {
    return nullptr;
  }

 private:
  Color color;
  Matrix4x4 transform;
};

} // namespace DynamicScene
} // namespace CMU462

#endif //CMU462_DYNAMICSCENE_SPOTLIGHT_H
