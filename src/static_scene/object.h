#ifndef CMU462_STATICSCENE_OBJECT_H
#define CMU462_STATICSCENE_OBJECT_H

#include "../halfEdgeMesh.h"
#include "scene.h"

namespace CMU462 { namespace StaticScene {

/**
 * A triangle mesh object.
 */
class Mesh : public SceneObject {
 public:

  /**
   * Constructor.
   * Construct a static mesh for rendering from halfedge mesh used in editing.
   * Note that this converts the input halfedge mesh into a collection of
   * world-space triangle primitives.
   */
  Mesh(const HalfedgeMesh& mesh);

  /**
   * Get all the primitives (Triangle) in the mesh.
   * Note that Triangle reference the mesh for the actual data.
   * \return all the primitives in the mesh
   */
  vector<Primitive*> get_primitives() const;

  /**
   * Get the BRDF of the surface materail of the mesh.
   * \return BRDF of the surface materail of the mesh
   */
  BRDF* get_brdf() const;

  // NOTE (sky):
  // Vertex and attribute arrays are made public so that a Triangle
  // primitive can index back into the mesh data without having to store
  // a pointer for each array.

  Vector3D *positions;  ///< position array
  Vector3D *normals;    ///< normal array

 private:

   BRDF* brdf; ///< BRDF of surface material

   vector<size_t> indices;  ///< triangles defined by indices

};

/**
 * A sphere object.
 */
class SphereObject : public SceneObject {
 public:

  /**
  * Constructor.
  * Construct a static mesh for rendering from halfedge mesh used in editing.
  * Note that this converts the input halfedge mesh into a collection of
  * world-space triangle primitives.
  */
  SphereObject(const Vector3D& o, double r);

  /**
  * Get all the primitives (Triangle) in the mesh.
  * Note that Triangle reference the mesh for the actual data.
  * \return all the primitives in the mesh
  */
  std::vector<Primitive*> get_primitives() const;

  /**
   * Get the BRDF of the surface material of the sphere.
   * \return BRDF of the surface material of the sphere
   */
  BRDF* get_brdf() const;

private:

  BRDF* brdf; ///< BRDF of the sphere objects' surface material

  Vector3D o; ///< origin
  double r;   ///< radius

}; // class SphereObject


} // namespace StaticScene
} // namespace CMU462

#endif // CMU462_STATICSCENE_OBJECT_H
