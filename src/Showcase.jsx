import { useEffect, useRef, useState } from 'react'
import { scenes, SCENE_SCROLL_VH } from './scenes'

const lerp = (a, b, t) => a + (b - a) * t
const clamp = (v, min, max) => Math.min(max, Math.max(min, v))

export default function Showcase() {
  const layerRefs = useRef([])
  const [active, setActive] = useState(0)

  useEffect(() => {
    let raf = null

    const update = () => {
      const vh = window.innerHeight
      const total = scenes.length * (SCENE_SCROLL_VH / 100) * vh
      const max = Math.max(total - vh, 1)
      // p goes from 0 -> scenes.length across the whole page
      const p = clamp(window.scrollY / max, 0, 1) * scenes.length

      scenes.forEach((s, i) => {
        const el = layerRefs.current[i]
        if (!el) return
        const localP = clamp(p - i, 0, 1) // progress within this shot
        const scale = lerp(s.start.scale, s.end.scale, localP)
        const x = lerp(s.start.x, s.end.x, localP)
        const y = lerp(s.start.y, s.end.y, localP)
        el.style.transform = `translate(${x}%, ${y}%) scale(${scale})`
        // crossfade: full at the shot's centre, fading at the edges
        el.style.opacity = clamp(1 - Math.abs(p - (i + 0.5)) / 0.65, 0, 1)
      })

      setActive(clamp(Math.round(p - 0.5), 0, scenes.length - 1))
      raf = null
    }

    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(update)
    }

    update()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
    }
  }, [])

  return (
    <>
      {/* the fixed cinematic stage */}
      <div className="stage">
        {scenes.map((s, i) => (
          <div
            key={i}
            ref={(el) => (layerRefs.current[i] = el)}
            className="layer"
            style={{ backgroundImage: `url(${s.image})` }}
          />
        ))}

        <div className="vignette" />

        <div className="overlay" key={active}>
          <p className="kicker">{scenes[active].kicker}</p>
          <h2 className="title">{scenes[active].title}</h2>
          <p className="sub">{scenes[active].sub}</p>
        </div>

        <div className="progress">
          {scenes.map((_, i) => (
            <span key={i} className={i === active ? 'dot on' : 'dot'} />
          ))}
        </div>
      </div>

      {/* invisible spacer that creates the scroll distance */}
      <div style={{ height: `${scenes.length * SCENE_SCROLL_VH}vh` }} />
    </>
  )
}
