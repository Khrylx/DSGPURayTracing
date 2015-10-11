#ifndef CMU462_DYNAMICSCENE_AMBIENTLIGHT_H
#define CMU462_DYNAMICSCENE_AMBIENTLIGHT_H

#include "scene.h"

namespace CMU462 { namespace DynamicScene {

class AmbientLight : public SceneLight {
 public:
  AmbientLight(const Color& color) {
    this->color = color;
  }

  void opengl_init_light(GLenum lightIndex) const {
    glMatrixMode(GL_MODELVIEW);
    glLightfv(lightIndex, GL_AMBIENT, &color.r);
    glEnable(lightIndex);
  }

  StaticScene::SceneLight *get_static_light() const {
    return nullptr;
  }

 private:
  Color color;
};

} // namespace DynamicScene
} // namespace CMU462

#endif // CMU462_DYNAMICSCENE_AMBIENTLIGHT_H
