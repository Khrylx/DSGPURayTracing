#ifndef CMU462_INTERSECT_H
#define CMU462_INTERSECT_H

#include <vector>

#include "CMU462/vector3d.h"
#include "CMU462/spectrum.h"
#include "CMU462/misc.h"

#include "static_scene/brdf.h"

namespace CMU462 { namespace StaticScene {

class Primitive;

/**
 * A record of an intersection point which includes the time of intersection
 * and other information needed for shading
 */
struct Intersection {

  Intersection() : t (INF_D), primitive(NULL), brdf(NULL) { }

  double t;    ///< time of intersection

  const Primitive* primitive;  ///< the primitive intersected

  Vector3D n;  ///< normal at point of intersection

  BRDF* brdf; ///< BRDF of the surface at point of intersection

  // More to follow.
};

} // namespace StaticScene
} // namespace CMU462

#endif // CMU462_INTERSECT_H
