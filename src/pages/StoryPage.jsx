import { useMemo, useState } from 'react'
import { marked } from 'marked'
import './MinutesPage.css'
import './StoryPage.css'

// Minutes are a dated log; these four are the standing narrative that a reader
// (or a reviewer) should be able to get without reading the entries in order.
// `final-claims` comes first because it is the only one that says what we would
// actually put our name to - the others explain how we got there.
const rawFiles = import.meta.glob('../../docs/{final-claims,research-story,results-reconciliation,human-review,proposal-vs-delivered}.md', {
  eager: true,
  query: '?raw',
  import: 'default',
})

const ORDER = [
  { slug: 'final-claims', label: '최종 주장', blurb: '우리가 주장하는 것과 주장하지 않는 것' },
  { slug: 'research-story', label: '연구 이야기', blurb: '체스에서 시작해 지금까지, 한 번에 읽는 판' },
  { slug: 'results-reconciliation', label: '숫자 화해', blurb: '효용 격차가 다섯 번 달라진 이유' },
  { slug: 'human-review', label: '사람 검토', blurb: '29개 항목의 처분 — 논문을 막는 건 둘뿐' },
  { slug: 'proposal-vs-delivered', label: '제안서 대비', blurb: '무엇을 줄였고 대신 무엇을 얻었나' },
]

function stripFrontmatter(raw) {
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/)
  return match ? match[2] : raw
}

const docs = ORDER.map((entry) => {
  const path = Object.keys(rawFiles).find((p) => p.endsWith(`${entry.slug}.md`))
  return { ...entry, body: path ? stripFrontmatter(rawFiles[path]) : '' }
}).filter((entry) => entry.body)

marked.setOptions({ gfm: true, breaks: false })

export default function StoryPage() {
  const [active, setActive] = useState(ORDER[0].slug)
  const current = docs.find((entry) => entry.slug === active) || docs[0]
  const html = useMemo(() => (current ? marked.parse(current.body) : ''), [current])

  return (
    <div className="page minutes-page">
      <div className="section">
        <p className="section-label">Story</p>
        <h2 className="section-title">연구 이야기</h2>
        <p className="section-desc">
          회의록이 날짜별 일지라면, 이 페이지는 연구 전체의 줄기입니다.
          발표·보고에서 그대로 쓸 수 있도록 정리했습니다.
        </p>

        <div className="story-tabs" role="tablist">
          {docs.map((entry) => (
            <button
              key={entry.slug}
              type="button"
              role="tab"
              aria-selected={entry.slug === active}
              className={`story-tab${entry.slug === active ? ' is-active' : ''}`}
              onClick={() => setActive(entry.slug)}
            >
              <span className="story-tab-label">{entry.label}</span>
              <span className="story-tab-blurb">{entry.blurb}</span>
            </button>
          ))}
        </div>

        {current && (
          <article
            className="markdown-body story-body"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        )}
      </div>
    </div>
  )
}
