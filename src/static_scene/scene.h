#ifndef CMU462_STATICSCENE_SCENE_H
#define CMU462_STATICSCENE_SCENE_H

#include "CMU462/CMU462.h"
#include "primitive.h"

#include <vector>

namespace CMU462 { namespace StaticScene {

/**
 * Interface for objects in the scene.
 */
class SceneObject {
 public:

  /**
   * Get all the primitives in the scene object.
   * \return a vector of all the primitives in the scene object
   */
  virtual std::vector<Primitive*> get_primitives() const = 0;

  /**
   * Get the surface BRDF of the object's surface.
   * \return the BRDF of the objects's surface
   */
  virtual BRDF* get_brdf() const = 0;

};


/**
 * Interface for lights in the scene.
 */
class SceneLight {
 public:

  /**
   * From a given point, returns a vector to a point on the light, as well as
   * the distance
   */
  virtual Vector3D sample(const Vector3D& p, double *distPtr) = 0;


  // More to follow (get_power_at_distance, etc).
};


/**
 * Represents a scene in a raytracer-friendly format. To speed up raytracing,
 * all data is already transformed to world space.
 */
struct Scene {
  Scene(const std::vector<SceneObject *>& objects,
        const std::vector<SceneLight *>& lights)
    : objects(objects), lights(lights) { }

  // kept to make sure they don't get deleted, in case the
  //  primitives depend on them (e.g. Mesh Triangles).
  std::vector<SceneObject*> objects;

  // for sake of consistency of the scene object Interface
  std::vector<SceneLight*> lights;

};

} // namespace StaticScene
} // namespace CMU462

#endif //CMU462_STATICSCENE_SCENE_H
