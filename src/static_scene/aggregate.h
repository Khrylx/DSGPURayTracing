#ifndef CMU462_STATICSCENE_AGGREGATE_H
#define CMU462_STATICSCENE_AGGREGATE_H

#include "scene.h"

namespace CMU462 { namespace StaticScene {

/**
 * Aggregate provides an interface for grouping multiple primitives together.
 * Because Aggretate itself implements the Primitive interface, no special
 * support is required for intersection acceleration. Integrators can be
 * written as if there was just a single Primitive in the scene, checking for
 * intersections without needing to be concerned about how they are actually
 * found. It is also easier to experiment new acceleration techniques by simply
 * adding a new Aggretate.
 */
class Aggregate : public Primitive {
 public:

  // Implements Primitive //

  // NOTE (sky):
  // There is no restrictions on how an Aggretate should be implemented but
  // normally an Aggretate should keep track of the primitives that it is
  // holding together. Note that during a ray - aggretate intersection, if
  // intersection information is to be updated (intersect2), the aggretate
  // implementation should store the address of the primitive that the ray
  // intersected as oppose to the aggretate itself.

	std::vector<Primitive*> primitives; ///< primitives enclosed in the aggregate

};


} // namespace StaticScene
} // namespace CMU462

#endif //CMU462_STATICSCENE_AGGREGATE_H
