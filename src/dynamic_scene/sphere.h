#ifndef CMU462_DYNAMICSCENE_SPHERE_H
#define CMU462_DYNAMICSCENE_SPHERE_H

#include "scene.h"
#include "material.h"

#include "../collada/sphere_info.h"

namespace CMU462 { namespace DynamicScene {

class Sphere : public SceneObject {
 public:
  Sphere(const Collada::SphereInfo& sphereInfo, const Vector3D& position,
         const double scale);

  void set_draw_styles(DrawStyle *defaultStyle, DrawStyle *hoveredStyle,
                       DrawStyle *selectedStyle);

  void render_in_opengl() const;

  BBox get_bbox();

  // All functions that are unused, because spheres can't be selected yet.
  double test_selection(const Vector2D& p, const Matrix4x4& worldTo3DH,
                        double minW) {
    return -1;
  }
  void confirm_hover() { }
  void confirm_select() { }
  void invalidate_hover() { }
  void invalidate_selection() { }
  SelectionInfo *get_selection_info() { return nullptr; }
  void drag_selection(float dx, float dy, const Matrix4x4& worldTo3DH) { }
  MeshView *get_mesh_view() { return nullptr; }

  StaticScene::SceneObject *get_static_object();

 private:
  Vector3D p;
  double r;
  DrawStyle *style;
  Material *material;
};

} // namespace DynamicScene
} // namespace CMU462

#endif //CMU462_DYNAMICSCENE_SPHERE_H
