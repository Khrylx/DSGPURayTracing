#ifndef CMU462_COLLADA_LIGHTINFO_H
#define CMU462_COLLADA_LIGHTINFO_H

#include "CMU462/color.h"

#include "collada_info.h"

namespace CMU462 { namespace Collada {

// For some reason enum classes aren't working for me here; not sure why.
namespace LightType {
  enum T {
    NONE,
    AMBIENT,
    DIRECTIONAL,
    POINT,
    SPOT
  };
}

class LightInfo : public Instance {
 public:
  LightType::T light_type; ///< type 

  Color color;          ///< color 

  Vector3D position;    ///< position
  Vector3D direction;   ///< direction

  float falloff_deg;    ///< degree of fall off angle
  float falloff_exp;    ///< fall out exponent

  float constant_att;   ///< constant attentuation factor  
  float linear_att;     ///< linear attentuation factor
  float quadratic_att;  ///< quadratic attentuation factor

  LightInfo();

}; // struct Light

std::ostream& operator<<( std::ostream& os, const LightInfo& light );

} // namespace Collada
} // namespace CMU462

#endif // CMU462_COLLADA_LIGHTINFO_H
