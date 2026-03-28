import React, { useEffect, useRef, useState } from 'react';
import {
  Sparkles,
  Play,
  Pause,
  Hand,
  MessageSquare,
  Volume2,
  ArrowRight,
  Coffee,
  Briefcase,
  Stethoscope,
  CreditCard,
  Building2,
  GraduationCap,
  Globe,
} from 'lucide-react';
import { motion, AnimatePresence, useScroll, useTransform, useMotionValue, useSpring } from 'framer-motion';
import { useInView } from 'react-intersection-observer';

const CountUp = ({ end, duration = 1.6, suffix = '', prefix = '' }) => {
  const [count, setCount] = useState(0);
  const { current: start } = useRef(performance.now());

  useEffect(() => {
    let frame;
    const tick = (now) => {
      const progress = Math.min((now - start) / (duration * 1000), 1);
      setCount(Math.floor(progress * end));
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [end, duration, start]);

  return (
    <span>
      {prefix}
      {suffix === '%' || suffix === 's' ? end : count.toLocaleString()}
      {suffix}
    </span>
  );
};

const accentColors = [
  { name: 'sage', value: '#C8D5B9', light: '#E8EFE3' },
  { name: 'lavender', value: '#D4C5E2', light: '#EEEBF5' },
  { name: 'rose', value: '#F4D7DA', light: '#FBF0F1' },
  { name: 'peach', value: '#FADEC9', light: '#FDF3EB' },
  { name: 'sky', value: '#D1E3F0', light: '#ECF4FA' },
  { name: 'coral', value: '#FF9B82', light: '#FFE5DF' },
];

const stats = [
  { value: 151000, label: 'BSL Users', detail: 'Fluent British Sign Language users in the UK (BDA, 2024)', accent: 'sage', cols: 5, rows: 2, badge: 'Verified • BDA 2024' },
  { value: 11, suffix: 'M', label: 'Deaf/HoH People', detail: 'Total deaf and hard of hearing population (RNID, 2024)', accent: 'lavender', cols: 4, rows: 1 },
  { value: 249, suffix: 'B', label: 'Spending Power', detail: 'Combined purchasing power (Purple Pound Report, 2023)', accent: 'rose', cols: 3, rows: 1 },
  { value: 2, suffix: '%', label: 'Business Access', detail: 'UK businesses offering BSL support (estimated)', accent: 'peach', cols: 7, rows: 1, icon: '🏢' },
  { value: 40, suffix: '-60', unit: '/hour', label: 'Interpreter Cost', detail: 'Average BSL interpreter rates (NRCPD)', accent: 'sky', cols: 5, rows: 1 },
  { value: 0.8, suffix: 's', label: 'Translation Time', detail: 'Our average real-time translation speed', accent: 'coral', cols: 6, rows: 1 },
];

const industries = [
  {
    title: 'Retail & Hospitality',
    description: 'Orders, check-ins, and table service become effortless for BSL users.',
    icon: Coffee,
    accent: 'peach',
    span: 'md:col-span-2',
  },
  {
    title: 'Healthcare',
    description: 'Instant triage and consultations without waiting for interpreters.',
    icon: Stethoscope,
    accent: 'sage',
  },
  {
    title: 'Financial Services',
    description: 'Accessible banking, insurance, and advisory sessions in-branch.',
    icon: CreditCard,
    accent: 'sky',
  },
  {
    title: 'Public Services',
    description: 'Council offices, libraries, and transport hubs that feel welcoming.',
    icon: Building2,
    accent: 'lavender',
    span: 'md:col-span-2',
  },
  {
    title: 'Education',
    description: 'Campus life, lectures, and events that are instantly inclusive.',
    icon: GraduationCap,
    accent: 'rose',
  },
  {
    title: 'Employment',
    description: 'Interviews, onboarding, and daily meetings without barriers.',
    icon: Briefcase,
    accent: 'coral',
  },
];

const steps = [
  { title: 'Detection', detail: 'Hands tracked in real-time at 30fps', icon: Hand },
  { title: 'Translation', detail: 'Sign → text within 0.8s', icon: MessageSquare },
  { title: 'Voice', detail: 'Friendly voice output with ripples', icon: Volume2 },
];

const profiles = [
  {
    name: 'Daniel',
    age: 30,
    location: 'Bristol',
    role: 'Software Developer',
    thought: 'Can I just order a latte without typing it out?',
    ring: '#C8D5B9',
    bubble: '#E8EFE3',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=Daniel&backgroundColor=C8D5B9',
    badge: 'BSL User',
    challenges: [
      { icon: '🏢', title: 'At Work', desc: 'Team meetings happen without me. Transcripts come hours later.' },
      { icon: '☕', title: 'Coffee Shops', desc: 'I type my order while everyone else speaks. It takes 3x longer.' },
      { icon: '🏥', title: 'Healthcare', desc: 'Emergency situations are terrifying. No time to find an interpreter.' },
    ],
  },
  {
    name: 'Sarah',
    age: 45,
    location: 'Manchester',
    role: 'Retail Manager',
    thought: 'Team briefings move fast — keep me in the loop.',
    ring: '#D4C5E2',
    bubble: '#EEEBF5',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=Sarah&backgroundColor=D4C5E2',
    badge: 'Retail Leader',
    challenges: [
      { icon: '🛍️', title: 'Customer Service', desc: 'Complaints I can’t respond to in real time.' },
      { icon: '📢', title: 'Staff Briefings', desc: 'Miss crucial info that everyone else hears instantly.' },
      { icon: '📞', title: 'Phone Calls', desc: 'Need everything in writing or sign — slows me down.' },
    ],
  },
  {
    name: 'James',
    age: 22,
    location: 'London',
    role: 'Student',
    thought: 'No interpreter on campus? Gesture has me covered.',
    ring: '#D1E3F0',
    bubble: '#ECF4FA',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=James&backgroundColor=D1E3F0',
    badge: 'Student',
    challenges: [
      { icon: '📚', title: 'Lectures', desc: 'Lectures with no interpreter become guesswork.' },
      { icon: '🎉', title: 'Social', desc: 'Events are tough to navigate without support.' },
      { icon: '💼', title: 'Interviews', desc: 'Interviews feel awkward without natural communication.' },
    ],
  },
];

const FloatingBlob = ({ delay = 0, gradient }) => (
  <motion.div
    className="absolute inset-0 opacity-30"
    style={{ background: gradient }}
    animate={{ scale: [1, 1.08, 1], rotate: [0, 6, -4, 0] }}
    transition={{ duration: 10, repeat: Infinity, delay }}
  />
);

// Smooth background + opacity transitions between sections.
const AppleStyleSection = ({ children, bgColor, nextBgColor, id }) => {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start end', 'end start'] });
  const backgroundColor = useTransform(scrollYProgress, [0, 1], [bgColor, nextBgColor]);
  const opacity = useTransform(scrollYProgress, [0, 0.5, 1], [1, 0.9, 1]);

  return (
    <motion.section
      id={id}
      ref={ref}
      style={{ backgroundColor, opacity }}
      className="relative min-h-screen overflow-hidden transition-all duration-1000"
    >
      {children}
    </motion.section>
  );
};

// Fade + slide content on scroll.
const FadeInSection = ({ children, delay = 0 }) => {
  const [ref, inView] = useInView({ threshold: 0.3, triggerOnce: false });
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 60 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 60 }}
      transition={{ duration: 0.8, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
};

// Scroll-linked movement.
const ScrollLinkedElement = ({ children }) => {
  const { scrollYProgress } = useScroll();
  const y = useTransform(scrollYProgress, [0, 1], [0, -200]);
  const scale = useTransform(scrollYProgress, [0, 0.5, 1], [1, 1.05, 1]);
  const opacity = useTransform(scrollYProgress, [0, 0.5, 1], [1, 0.8, 1]);
  return (
    <motion.div style={{ y, scale, opacity, willChange: 'transform' }}>
      {children}
    </motion.div>
  );
};

const AnimatedNumber = ({ end, duration = 1.6, suffix = '' }) => {
  const [value, setValue] = useState(0);
  const [ref, inView] = useInView({ threshold: 0.5, triggerOnce: true });

  useEffect(() => {
    if (!inView) return;
    let start;
    const step = (ts) => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / (duration * 1000), 1);
      setValue(Math.floor(end * progress));
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [inView, end, duration]);

  return (
    <span ref={ref}>
      {value.toLocaleString()}
      {suffix}
    </span>
  );
};

// Magnetic hover + tilt wrapper for stat cards.
const TiltStatCard = ({ children }) => {
  const [rotateX, setRotateX] = useState(0);
  const [rotateY, setRotateY] = useState(0);
  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    setRotateX((y - 0.5) * -12);
    setRotateY((x - 0.5) * 12);
  };
  const handleMouseLeave = () => {
    setRotateX(0);
    setRotateY(0);
  };
  return (
    <motion.div
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      animate={{ rotateX, rotateY }}
      transition={{ type: 'spring', stiffness: 250, damping: 18 }}
      style={{ transformStyle: 'preserve-3d', perspective: 900, willChange: 'transform' }}
      className="relative"
    >
      <div className="relative" style={{ transform: 'translateZ(24px)' }}>
        {children}
      </div>
    </motion.div>
  );
};

const MagneticStatCard = ({ stat, label, detail, accent }) => {
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const springX = useSpring(x, { damping: 15, stiffness: 150 });
  const springY = useSpring(y, { damping: 15, stiffness: 150 });

  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    x.set((e.clientX - centerX) * 0.08);
    y.set((e.clientY - centerY) * 0.08);
  };

  const handleMouseLeave = () => {
    x.set(0);
    y.set(0);
  };

  return (
    <motion.div
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{ x: springX, y: springY, willChange: 'transform' }}
      className="relative bg-white/70 backdrop-blur-xl rounded-3xl p-8 border-t-4 shadow-xl cursor-pointer overflow-hidden"
    >
      <MorphingBackground colors={[accent.light, accent.value]} />
      <div className="relative space-y-3">
        <div className="text-6xl md:text-7xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
          <AnimatedNumber end={stat.value} suffix={stat.suffix || ''} />
        </div>
        <h3 className="text-2xl font-semibold">{label}</h3>
        <p className="text-[var(--text-secondary)]">{detail}</p>
        {stat.badge && (
          <div className="inline-block bg-white/70 px-4 py-2 rounded-full text-sm font-medium border border-white">
            {stat.badge}
          </div>
        )}
      </div>
    </motion.div>
  );
};

const MorphingBackground = ({ colors }) => (
  <div className="absolute inset-0 opacity-30 rounded-3xl pointer-events-none overflow-hidden">
    <svg className="w-full h-full">
      <defs>
        <linearGradient id="morphGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor={colors[0]} />
          <stop offset="100%" stopColor={colors[1]} />
        </linearGradient>
      </defs>
      <motion.path
        d="M0,50 Q25,25 50,50 T100,50 L100,100 L0,100 Z"
        fill="url(#morphGradient)"
        animate={{
          d: [
            'M0,50 Q25,25 50,50 T100,50 L100,100 L0,100 Z',
            'M0,70 Q25,50 50,70 T100,70 L100,100 L0,100 Z',
            'M0,50 Q25,25 50,50 T100,50 L100,100 L0,100 Z',
          ],
        }}
        transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
      />
    </svg>
  </div>
);

const StatsGrid = ({ stats }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-12 auto-rows-[1fr] gap-6">
      {stats.map((stat, i) => {
        const accent = accentColors.find((c) => c.name === stat.accent) || accentColors[0];
        const gridCol = stat.cols || 4; // spans within the 12-col layout
        const gridRow = stat.rows || 1;
        return (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, scale: 0.9, y: 30 }}
            whileInView={{ opacity: 1, scale: 1, y: 0 }}
            viewport={{ once: true, margin: '-80px' }}
            transition={{ duration: 0.6, delay: i * 0.1, ease: [0.22, 1, 0.36, 1] }}
            className="col-span-12 md:col-span-6"
            style={{
              willChange: 'transform',
              gridColumn: `span ${gridCol} / span ${gridCol}`,
              gridRow: `span ${gridRow} / span ${gridRow}`,
            }}
          >
            <TiltStatCard>
              <MagneticStatCard stat={stat} label={stat.label} detail={stat.detail} accent={accent} />
            </TiltStatCard>
          </motion.div>
        );
      })}
    </div>
  );
};

const CreativeRunningLine = () => {
  const items = [
    { icon: '🤝', text: 'Only 2% offer BSL', color: '#C8D5B9' },
    { icon: '💷', text: '£249B market', color: '#D4C5E2' },
    { icon: '⏰', text: '£40-60/hour interpreters', color: '#F4D7DA' },
    { icon: '👥', text: '151,000 BSL users', color: '#FADEC9' },
    { icon: '💡', text: '0.8s translation', color: '#D1E3F0' },
  ];

  return (
    <div className="relative overflow-hidden bg-gradient-to-r from-[#FEFDFB] via-[#F9F7F4] to-[#FEFDFB] py-8 border-y-2 border-orange-100">
      <motion.div
        className="absolute inset-0"
        style={{ background: 'linear-gradient(90deg, transparent, rgba(255,155,130,0.12), transparent)' }}
        animate={{ x: ['-100%', '100%'] }}
        transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
      />
      <motion.div
        className="flex gap-12 whitespace-nowrap px-6"
        animate={{ x: ['0%', '-50%'] }}
        transition={{ duration: 28, repeat: Infinity, ease: 'linear' }}
      >
        {[...items, ...items].map((item, i) => (
          <div
            key={i}
            className="flex items-center gap-4 px-6 py-3 rounded-full bg-white/70 backdrop-blur-sm border border-gray-100 shadow-lg"
            style={{ borderTop: `3px solid ${item.color}` }}
          >
            <motion.span
              className="text-4xl"
              animate={{ rotate: [0, 10, -10, 0], scale: [1, 1.15, 1] }}
              transition={{ duration: 2, repeat: Infinity, delay: i * 0.15 }}
            >
              {item.icon}
            </motion.span>
            <span
              className="text-2xl font-bold"
              style={{
                fontFamily: "'Outfit', sans-serif",
                background: `linear-gradient(135deg, #2C2825, ${item.color})`,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              {item.text}
            </span>
          </div>
        ))}
      </motion.div>
    </div>
  );
};

const GestureProApp = () => {
  const [scrolled, setScrolled] = useState(false);
  const [showFeed, setShowFeed] = useState(false);
  const [feedUrl, setFeedUrl] = useState('');
  const [activeProfile, setActiveProfile] = useState(0);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 120);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const startLiveFeed = () => {
    // Bust cache so each start reopens the MJPEG stream.
    setFeedUrl(`/video_feed?ts=${Date.now()}`);
    setShowFeed(true);
  };

  const stopLiveFeed = async () => {
    setShowFeed(false);
    setFeedUrl('');
    try {
      await fetch('/stop_infer', { method: 'POST' });
    } catch (e) {
      console.error('Failed to stop inference', e);
    }
  };

  const currentProfile = profiles[activeProfile];

  const particleOffsets = [
    { x: -80, y: -20 },
    { x: 40, y: -60 },
    { x: 120, y: 10 },
    { x: -30, y: -100 },
    { x: 90, y: -90 },
  ];

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)] relative overflow-hidden">
      {/* Navigation behavior */}
      <AnimatePresence>
        {!scrolled ? (
          <motion.nav
            key="top"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="absolute top-0 left-0 w-full z-50"
          >
            <div className="container flex justify-between items-center py-8">
              <div className="flex items-center gap-2">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[var(--brand-coral)] to-[var(--brand-peach)] flex items-center justify-center shadow-[var(--shadow-coral)]">
                  <Sparkles className="w-5 h-5 text-white" />
                </div>
                <span className="text-2xl font-bold">Gesture</span>
              </div>
              <div className="hidden md:flex gap-12 text-lg text-[var(--text-secondary)]">
                <a href="#stories" className="hover:text-[var(--accent-orange)]">Stories</a>
                <a href="#how" className="hover:text-[var(--accent-orange)]">How it Works</a>
                <a href="#impact" className="hover:text-[var(--accent-orange)]">Impact</a>
              </div>
              <button className="bg-gradient-to-r from-[var(--brand-coral)] to-[var(--brand-apricot)] text-white px-6 py-3 rounded-full font-semibold shadow-[0_12px_32px_rgba(255,155,130,0.20)]">
                Try Now
              </button>
            </div>
          </motion.nav>
        ) : (
          <motion.nav
            key="floating"
            initial={{ y: -80, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed top-6 left-1/2 -translate-x-1/2 z-50"
          >
            <div className="bg-white/95 backdrop-blur-xl rounded-full px-8 py-3 shadow-xl flex items-center gap-8 border border-[rgba(200,213,185,0.3)]">
              <span className="font-bold">Gesture</span>
              <div className="hidden md:flex gap-6 text-sm text-[var(--text-secondary)]">
                <a href="#stories" className="hover:text-[var(--accent-orange)]">Stories</a>
                <a href="#how" className="hover:text-[var(--accent-orange)]">How it Works</a>
                <a href="#impact" className="hover:text-[var(--accent-orange)]">Impact</a>
              </div>
              <button className="bg-gradient-to-r from-[var(--brand-coral)] to-[var(--brand-apricot)] text-white px-6 py-2 rounded-full text-sm font-semibold shadow-[0_12px_32px_rgba(255,155,130,0.20)]">
                Try Now
              </button>
            </div>
          </motion.nav>
        )}
      </AnimatePresence>

      {/* Live camera only with parallax */}
      <AppleStyleSection id="stories" bgColor="#FFF9F0" nextBgColor="#FFF5E8">
        <div className="absolute inset-0">
          {[
            { color: 'rgba(200, 213, 185, 0.15)', size: 600, delay: 0 },
            { color: 'rgba(212, 197, 226, 0.15)', size: 500, delay: 2 },
            { color: 'rgba(244, 215, 218, 0.15)', size: 550, delay: 4 },
          ].map((blob, i) => (
            <motion.div
              key={i}
              className="absolute rounded-full blur-3xl"
              style={{ width: blob.size, height: blob.size, background: blob.color }}
              animate={{ x: ['20%', '80%', '20%'], y: ['20%', '60%', '20%'] }}
              transition={{ duration: 20 + i * 5, repeat: Infinity, ease: 'easeInOut', delay: blob.delay }}
            />
          ))}
        </div>

        <div className="relative z-10 pt-32 pb-20">
          <div className="container">
            <ScrollLinkedElement>
              <FadeInSection>
                <div className="text-center mb-10">
                  <div className="inline-flex items-center gap-2 bg-white/60 backdrop-blur-sm px-6 py-3 rounded-full mb-6 border border-gray-200">
                    <span className="text-sm font-medium text-gray-700">✨ Live interpreter</span>
                  </div>
                  <h1 className="text-5xl md:text-6xl font-bold leading-tight mb-4">
                    Start your live gesture view
                  </h1>
                  <p className="text-xl text-[var(--text-secondary)] max-w-3xl mx-auto">
                    Open the camera feed and run real-time inference powered by infer.py.
                  </p>
                </div>
              </FadeInSection>

              <FadeInSection delay={0.1}>
                <div className="max-w-5xl mx-auto glass-card rounded-3xl border border-[rgba(44,40,37,0.08)] p-6 shadow-xl">
                  <div className="flex flex-wrap items-center justify-center gap-3 mb-4">
                    <motion.button
                      whileHover={{ scale: 1.05, y: -2, boxShadow: '0 20px 48px rgba(255,155,130,0.35)' }}
                      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                      onClick={startLiveFeed}
                      className="flex items-center gap-2 bg-gradient-to-r from-[var(--brand-coral)] to-[var(--brand-apricot)] text-white px-6 py-3 rounded-full font-semibold shadow-[0_16px_32px_rgba(255,155,130,0.25)]"
                    >
                      {showFeed ? <span>Restart Feed</span> : <><Play className="w-5 h-5" /> <span>Start Live View</span></>}
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.05, y: -2 }}
                      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                      onClick={stopLiveFeed}
                      className="flex items-center gap-2 px-6 py-3 rounded-full font-semibold border border-[rgba(44,40,37,0.12)] text-[var(--text-primary)] bg-white hover:bg-[var(--surface-cream)] transition-all"
                    >
                      <Pause className="w-5 h-5" />
                      Stop
                    </motion.button>
                  </div>
                  <div className="relative overflow-hidden rounded-2xl border border-[rgba(44,40,37,0.06)] bg-white/60 min-h-[320px] flex items-center justify-center">
                    {showFeed ? (
                      <img
                        src={feedUrl}
                        alt="Live gesture feed"
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <div className="text-center space-y-2 text-[var(--text-secondary)]">
                        <span role="img" aria-label="wave">👋</span>
                        <div className="text-lg font-semibold text-[var(--text-primary)]">Click “Start Live View” to open the camera feed.</div>
                        <div>Keep this tab active to maintain the stream.</div>
                      </div>
                    )}
                  </div>
                </div>
              </FadeInSection>
            </ScrollLinkedElement>
          </div>
        </div>
      </AppleStyleSection>

      <CreativeRunningLine />

      {/* Customer story */}
      <AppleStyleSection bgColor="#F9F7F4" nextBgColor="#FEFDFB">
        <div className="container section-block">
          <FadeInSection>
            <div className="grid lg:grid-cols-2 gap-20 items-center">
              <div className="relative">
                <div className="aspect-square rounded-3xl overflow-hidden bg-gradient-to-br from-[var(--accent-sage)] to-[var(--accent-lavender)] relative shadow-2xl">
                  <img src={currentProfile.avatar} alt={currentProfile.name} className="w-full h-full object-cover transition-transform duration-500 will-change-transform hover:scale-105" />
                  <div className="absolute top-6 right-6 bg-white/90 backdrop-blur-sm px-4 py-2 rounded-full text-sm font-medium text-gray-700">
                    {currentProfile.badge}
                  </div>
                </div>
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.2 }}
                  className="absolute -bottom-10 -left-8 bg-white rounded-3xl p-6 shadow-2xl max-w-xs"
                >
                  <p className="text-lg italic text-gray-700">
                    "{currentProfile.thought}"
                  </p>
                </motion.div>
              </div>

              <div className="space-y-8">
                <div>
                  <div className="inline-flex items-center gap-2 bg-[var(--accent-sage)]/20 px-4 py-2 rounded-full mb-4">
                    <span className="text-2xl">👋</span>
                    <span className="text-sm font-medium text-gray-700">Meet our community</span>
                  </div>
                  <h2 className="text-5xl font-bold mb-3">Hi, I'm {currentProfile.name}</h2>
                  <p className="text-xl text-gray-600 mb-6">
                    {currentProfile.age} • {currentProfile.location} • {currentProfile.role}
                  </p>
                </div>

                <div className="space-y-4">
                  <h3 className="text-2xl font-semibold text-gray-800">Daily challenges I face:</h3>
                  {currentProfile.challenges.map((problem, i) => (
                    <FadeInSection key={problem.title} delay={i * 0.05}>
                      <div className="flex gap-4 p-6 bg-white/60 backdrop-blur-sm rounded-2xl border border-gray-100 hover:-translate-y-2 transition-all hover:shadow-xl">
                        <div className="text-3xl flex-shrink-0">{problem.icon}</div>
                        <div>
                          <h4 className="font-semibold text-lg mb-1">{problem.title}</h4>
                          <p className="text-gray-600">{problem.desc}</p>
                        </div>
                      </div>
                    </FadeInSection>
                  ))}
                </div>

                <div className="flex items-center gap-3">
                  {profiles.map((profile, i) => (
                    <button
                      key={profile.name}
                      onClick={() => setActiveProfile(i)}
                      className={`w-3 h-3 rounded-full transition-all ${i === activeProfile ? 'w-6 bg-[var(--brand-coral)]' : 'bg-[var(--text-muted)]'}`}
                      aria-label={`Profile ${i + 1}`}
                    />
                  ))}
                </div>

                <motion.button
                  whileHover={{ scale: 1.05, y: -2 }}
                  transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                  className="flex items-center gap-3 bg-gradient-to-r from-[var(--brand-coral)] to-[var(--brand-apricot)] text-white px-8 py-4 rounded-full font-semibold text-lg shadow-xl hover:shadow-2xl transition-all"
                >
                  <Play className="w-6 h-6" />
                  Hear {currentProfile.name}'s full story
                </motion.button>
              </div>
            </div>
          </FadeInSection>
        </div>
      </AppleStyleSection>

      {/* How it works */}
      <AppleStyleSection id="how" bgColor="#FEFDFB" nextBgColor="#F9F7F4">
        <div className="container section-block">
          <div className="flex flex-col md:flex-row gap-14 items-center">
            <FadeInSection>
              <div className="flex-1 space-y-4">
                <div className="inline-flex px-4 py-2 rounded-full bg-[var(--surface-pearl)] border border-[rgba(212,197,226,0.5)] text-[var(--text-secondary)] text-sm">
                  How Gesture feels
                </div>
                <h2 className="text-4xl md:text-5xl font-extrabold leading-tight">
                  Warm, familiar steps
                  <span className="block text-[var(--text-secondary)]">that mirror a natural chat.</span>
                </h2>
                <p className="text-lg text-[var(--text-secondary)] max-w-xl">
                  From hand detection to voice output, each micro-interaction is designed to feel delightful and human-first.
                </p>
                <div className="flex items-center gap-3 text-[var(--text-secondary)]">
                  <Sparkles className="w-5 h-5 text-[var(--brand-coral)]" />
                  Organic blobs, hand-drawn circles, and friendly ripples guide the experience.
                </div>
              </div>
            </FadeInSection>

            <div className="flex-1 space-y-6">
              {steps.map((step, idx) => (
                <FadeInSection key={step.title} delay={idx * 0.08}>
                  <div className="relative p-5 rounded-2xl glass-card border border-[rgba(255,155,130,0.2)] hover:-translate-y-2 transition-all">
                    <div className="absolute -left-3 -top-3 w-10 h-10 rounded-full bg-[var(--accent-peach)]/70 text-[var(--text-primary)] font-bold flex items-center justify-center">
                      0{idx + 1}
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 rounded-xl bg-[var(--bg-accent)] flex items-center justify-center text-[var(--accent-orange)]">
                        <step.icon className="w-6 h-6" />
                      </div>
                      <div>
                        <h4 className="text-xl font-semibold">{step.title}</h4>
                        <p className="text-[var(--text-secondary)]">{step.detail}</p>
                      </div>
                    </div>
                  </div>
                </FadeInSection>
              ))}
            </div>
          </div>
        </div>
      </AppleStyleSection>

      {/* Impact / stats */}
      <AppleStyleSection id="impact" bgColor="#F9F7F4" nextBgColor="#FEFDFB">
        <div className="container section-block">
          <FadeInSection>
            <div className="text-center mb-16 max-w-3xl mx-auto">
              <h2 className="text-5xl md:text-6xl font-extrabold mb-4">
                Gesture impact, at a glance
              </h2>
              <p className="text-xl text-[var(--text-secondary)]">
                Gesture pairs premium craft with measurable outcomes. Each card highlights why warm accessibility matters.
              </p>
            </div>
          </FadeInSection>
          <StatsGrid stats={stats} />
        </div>
      </AppleStyleSection>

      {/* Industries */}
      <section className="section-block bg-[var(--surface-cream)]">
        <div className="container">
          <div className="flex items-center justify-between gap-4 mb-10">
            <div>
              <h2 className="text-4xl md:text-5xl font-extrabold">Built for every industry</h2>
              <p className="text-lg text-[var(--text-secondary)] mt-2">Bento-style cards with playful lifts and premium accents.</p>
            </div>
            <div className="hidden md:flex items-center gap-3 text-[var(--text-secondary)]">
              <Globe className="w-5 h-5 text-[var(--accent-orange)]" />
              Always-on accessibility, everywhere.
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 auto-rows-[1fr]">
            {industries.map((item, idx) => {
              const accent = accentColors.find((c) => c.name === item.accent) || accentColors[0];
              const Icon = item.icon;
              return (
                <motion.div
                  key={item.title}
                  className={`rounded-3xl p-6 bg-white shadow-lg border border-[rgba(44,40,37,0.06)] ${item.span || ''} flex flex-col justify-between`}
                  whileHover={{ y: -8, boxShadow: '0 20px 48px rgba(44,40,37,0.12)' }}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: idx * 0.05 }}
                >
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 rounded-xl flex items-center justify-center" style={{ background: accent.light }}>
                      <Icon className="w-6 h-6" style={{ color: accent.value }} />
                    </div>
                    <h3 className="text-xl font-bold">{item.title}</h3>
                  </div>
                  <p className="text-[var(--text-secondary)] mt-4 flex-1">{item.description}</p>
                  <button className="mt-6 inline-flex items-center gap-2 text-[var(--accent-orange)] font-semibold">
                    See it in action <ArrowRight className="w-4 h-4" />
                  </button>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Footer CTA */}
      <section className="section-block bg-gradient-to-b from-[var(--surface-pearl)] to-[var(--bg-primary)]">
        <div className="container max-w-4xl mx-auto text-center">
          <h2 className="text-4xl md:text-5xl font-extrabold mb-4">Ready to feel heard?</h2>
          <p className="text-lg text-[var(--text-secondary)] mb-8">
            Gesture brings warmth back to accessibility with playful visuals and premium craft.
          </p>
          <div className="flex justify-center gap-4">
            <button className="px-10 py-4 bg-gradient-to-r from-[var(--brand-coral)] to-[var(--brand-apricot)] text-white rounded-full font-semibold shadow-[0_12px_32px_rgba(255,155,130,0.20)]">
              Request Early Access
            </button>
            <button className="px-10 py-4 rounded-full border border-[rgba(44,40,37,0.1)] text-[var(--text-primary)] bg-white shadow-md">
              View Demo
            </button>
          </div>
        </div>
      </section>

      <footer className="border-t border-[rgba(44,40,37,0.08)] py-8 px-6 md:px-10 bg-white">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[var(--brand-coral)] to-[var(--brand-peach)] flex items-center justify-center shadow-[var(--shadow-coral)]">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-[var(--text-primary)]">Gesture</span>
          </div>
          <div className="text-[var(--text-secondary)] text-sm">© 2024 Gesture. Crafted with warmth and intent.</div>
        </div>
      </footer>
    </div>
  );
};

export default GestureProApp;
