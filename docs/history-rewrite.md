# History rewrite (2026-10-23)

On Oct 23 the published history was rewritten to remove files that only served the Claude Code and Codex workflow:
`CLAUDE.md`, `AGENTS.md`, `docs/plan.md`, `docs/plans/`, `docs/handover/` and `docs/reviews/` (D86). Every commit ID
changed. The final tree did not: the newest commit's files are byte-identical before and after. Eight commits that only
touched the removed files are gone.

Older documents, run records and reports keep the IDs they were written with. Evaluation provenance does not depend on
them: each run manifest also records a `source_digest`, a hash of the source files themselves, which the rewrite left
unchanged. This table maps every ID quoted in the published docs and reports to its new commit.

| Old ID | New ID | Where it is quoted |
|---|---|---|
| `32cbd6d` | `5ad96ea` | the frozen source of the final evaluation (D68, D70) |
| `22f7bd2` | `fc4fc16` | `git.head` of the D70 runs (`docs/results/metrics.json`, `report.html`) |
| `0e4d54b` | `9d1c918` | the source of the Ultra runs (D82, `docs/results/ultra/`) |
| `020bafb` | `2e007b4` | the source of the D85 runs (`docs/results/a1/`); the `submission-data` tag |
| `2531621` | `c4f9fd1` | quoted in `docs/decisions.md` |
| `ea4caac` | `0464216` | quoted in `docs/decisions.md` |
| `81d3839` | `f78139a` | quoted in `docs/decisions.md` |
| `dc231a1` | `d76f3f3` | quoted in `docs/decisions.md` |
| `dcb5d7b` | `06bdeff` | quoted in `docs/decisions.md` |
| `d90e968` | `04d1a73` | quoted in `docs/decisions.md` |
| `708235d` | `c57ca09` | quoted in `docs/decisions.md` |
| `ad54e18` | `f814ab6` | quoted in `docs/decisions.md` |
| `66827d4` | `5c2fa66` | quoted in `docs/decisions.md`; `git.head` in `docs/results/shadow/summary.json` |
| `e2cc655` | `12e3f68` | quoted in `docs/decisions.md` (D80) |
| `ea40098` | `fcbf517` | quoted in `docs/decisions.md` (D80) |
| `7c058f4` | `94e4d71` | `git.head` in `docs/evidence/sigma-calibration-dev.json` |

IDs quoted in `docs/decisions.md` that are not in this table (`5d2e22c`, `f223ce0`, `e85afa9`, `e1d4d92`, `de6bd4e`,
`835e9c6`) belonged to local working branches that were never published.

## Every commit

Old ID → new ID for all 252 published commits, oldest last.

```
1729ca8574 2405d86e5f
59e7fa0dd2 f1d03f94fc
81ccb93656 51a88704ae
742966d9f3 b088df3961
11d04992c0 16a361639a
1151cbd61e 1618f4545a
020bafb689 2e007b4b90
2721ce27f3 143c18d4e0
8f3f2fbb4c 22766c356b
19f17a7d29 d4473be206
6b640ab38f 57a646f77c
4009a6b65e 3ad2b63a4f
ea4009888a fcbf5173e2
e2cc655334 12e3f68ec2
0e4d54b479 9d1c9189f2
33a933bc90 f170ba5114
cfd23f81d2 3756a4f15b
06e35e3e03 f446be0859
b9980a58bb d12d2668d0
a29dea4f9b 26c1d03865
7f640de554 d14ad83f0c
89ab98f1f0 9e1f7b781f
50fa1b4d05 91958af186
d6d520c6f1 f5467fd1ef
fd836bdc2c 78f4077a20
6e7d938150 fbd829022c
ea3c775265 e1879500e2
60bc6d320f bdc406f8a5
ee84f7e3ee 1598499a7d
4485d80f8f 3339fec1ff
bfb26135f6 2ad1391aa1
5d60cddd23 3e9d29e1ae
2531621e84 c4f9fd17a8
15526db613 e125eb0ce2
54acb0fa6e 6320d58598
4e64309627 9806a39348
5ae2d7ffb7 1d94e2df1b
9bc240e53c ce488058ee
fe7f0cb5cb ee53ec3723
abb9130e42 a63279fd29
66827d44f4 5c2fa66e2e
d2d260e935 596867b48b
4fe9a19977 e26714ab46
05a6d808ba 46570c6cdf
77f0243daf 9e323bae59
681d94add5 18cc6387b7
bbc2ee9cc9 3353e3469a
b5de2e03aa 2e0246b465
a52484772c c267a28cad
fcb2d327ca b0a406066d
c933be7908 422a2d6f90
651564451e 5ec8028055
ad54e18723 f814ab6599
8ced9aa962 7e9d2816aa
c9638c47d7 6ef261d50b
2353e942d2 27d845d9f2
47d1b305be 8f9a456e5e
1614612dcf 10037455a0
22f7bd2f8b fc4fc16d9b
32cbd6dd0b 5ad96eaf2a
fe9b11c2e8 4bd34d658b
9c5c70da93 52d2036a6a
fc0cf39745 bcf1ee614a
4c60a2e6cb e9730d10e5
c82d390868 dce999bcc0
09ffbbb409 04f8c44b69
140ab9f8be 3995c7ae5d
e885156af5 8db1f67e94
7bce0ccf9c 4070bdf443
b90132bffd aec2cde34c
bb8fff3c5a b6a42b519a
723f317dc3 221a9a37a8
0dc435c595 a9958fdc70
5b24748f3d 7709311d19
7060b1ee57 252e278093
69a7fbada3 f67c207818
5e314e6890 74b06c2b06
c4361dd948 2632916b95
6722aa5d33 a684ff0b45
eca3c160f4 16af25b6af
124d770d85 8ee98c2971
1fdfb07f9b bf8a6ec015
d87cc1ae0c 1a7e746300
6a1701ba04 906b1dfb39
df8c72ff9e 6fcc3b662e
e7b86b95e1 7dc5aeed1a
d02d3bd267 dd0330fceb
b2195db9dd 59fee75478
feb55205c4 ec3be83401
78bad2c2aa e9b9df817b
ee14ca10b9 1d4f9494d0
3f84de073e eaf229bbf8
708235d6bf c57ca09d36
894d43b2a1 edc1be6cd6
992538624f 3e04a2d4e2
d9d05d9b8d f7ad500c71
e8b9132733 2470b96fbb
c988ba14be 1553db7b1b
2189faa49b 7d6e4ac738
d90e96841d 04d1a732aa
2b9089f7bf 8e779322bf
7c058f4a85 94e4d71db0
7c08b693ad 068b3743a0
59620f4cf4 4d70ee4d82
7d9a6aa3ff 3b499b46d7
fdb1adabee 5fd795f9f9
7f91a2d99c bfd7fdab83
fa44cd580b 667becabde
ba24508a18 72ca8cff22
43f16ebf24 7b7d76950f
5e024325a7 799259dd74
1dc5e871c7 f580a68f91
0208ee9485 61135d71b8
1877afc918 00aa4d87f6
268d5b0294 c20b963d8d
abcc1db516 ae44df1148
3283c292b0 ff6f636651
5a09b979c8 11ca4e12e9
143efe96d5 13ea9adbfa
51e0b96c00 45f924b8f9
c6fd4d796a c6c9796aa9
8bf9c2b35f 7292870ceb
a79070a7d1 ae5cf05748
7989eade83 8155eb9b12
f492d77ab9 1a104ba1fa
a4bec1c25e eb4ad9e175
7b9261ec7d 6b1da8834c
53cbcfd1c4 4d6b683196
9a0876eeaf 107030407d
431d8ed7ae 37195bb190
6586c646d8 9ef2fbf39e
2b72d57ec0 c6ebd7c024
9148e73ae9 dec7f51f48
95f569c331 e887af4c02
4c1ed1b812 8437e44a01
716aaec470 4743e730c1
827b60064a db19b81d2c
4c6a546915 2319a1e23f
3409c1eb2c c4ffa8e52d
8df250c088 15acea863d
2c7f40a621 41114887ad
ec24ae7f75 a78db340ad
6850b27369 a639d9b5e6
87fd019578 4352fe6ee4
fe62a914a9 694d19726a
87a26cbd21 c7b7ca9f68
e2d5ac34c5 2b5620ec9e
bbcfcadbb7 42cdc76284
2c68ed0413 fbf8c8564c
e49bde5384 d5572da960
80cc63fe33 e00169a76f
6ead9b4c58 fa9bd194cc
cd4539d49a 230a6b3961
12ae877ac2 ec2819015d
20d5db3f9a 90b1ceb93a
5f3b6b0090 6ce111bc8a
0133263477 0119db1b19
72c9f1bfed 947335ae33
f79f339b9b 781ae98b2f
39bd82598f 92f25e6822
492201bf9d 32e9ba8ea3
999d4a8980 9f03eaae25
24b984e7ec c5753a88b7
ef0dc7362f 6386cb14ac
c0e856664f 4c47904330
2912d83932 2212a47a56
087bcca893 98ccf6fa02
b0666f4a59 ec502657bd
d0921b7a54 cbb2f03b05
654bf688f2 89b790c621
f815dc755a 133b773c0e
573fe6240b fbcc8940a9
c9669a1755 1889360a2a
22f93c42be 70e5470dc3
6fe3759863 efdaaffdff
ac5e7f7402 985e12d198
82e354cf9d e452284738
10fb1f989c 23d82170a9
5b44ae2d54 fc11a2f192
676d99c0bf 80812d0ea5
dcb5d7bf2e 06bdeff317
eb3d043a60 df2d4e1ea6
73ac78942f 13b23b21dd
c026cb8a84 ba03b72191
e46eca0fc0 0435ec8e62
a778b0904f 2be28a9a2b
539c78f591 adb23b8453
2d4ca053bc 80e233878c
26576c9f48 7dec56ccf5
e6a5bd6a62 e0f053a0d3
806554e9aa 2126c3a88b
73020323a2 ea65b1bd92
c4ac5b740a 526576a0cf
e3c48c7776 7f207f7639
d7e7fa6a90 5754706f23
ee59178569 4204d9d84e
bd2773986d 93aab2bce4
9d7969e706 fd5cc1a23a
96bbbdc219 e3683529a4
28d71c1029 d7955f7129
518d3130d9 b8b88cf776
e02732820a 9e32f1a1c3
ae0ff73348 5011430da6
dc74d1f2df bce3eb451d
50acf47503 ed7a7f975e
28dee89095 fa078a494b
49c8f49928 32d151ec9e
87f13f5570 9def8489f7
eabf86d634 9e0a5855b7
bdb3f798eb be507f687e
deb6e6af3f 7d350fa75a
94451e7f6e ca50dbffd3
09d5d40ec2 1f70b05702
ef93bd925e 04d0813e36
73b5d8417a 5e77eb0d9d
23e7b605e2 c627c7dacc
dbecded44a 40c106051c
e94951fc36 35d859ff07
a6190306cd fa9df9fa02
98d974d0c5 df99e08a70
e9472188a6 0899888a57
9d7f2a2747 99922b76f8
0f5e128d41 4740626988
cf325b0544 9b120e59ab
621a1e845a 719f30d38a
afb10c3073 341e26d151
dc231a113c d76f3f3ca6
574a94157e e6448137e9
f703826c8c 6b3510a5ab
8b186bec72 4b774ed319
26a8490aff 03a399cbad
97bda863d4 2c50184797
b342d70372 80fb304175
4ee6c4dd86 f8d974ab95
ea4caac4fb 0464216778
92049d6cdb e067a2a07b
fa5449ae32 3512a97956
01061fd26e ab065e3b4a
7c474f2533 2cdc0c18a8
5e9b24e222 0090fcf692
048684bbcc 2d5f12dd1e
4a8086a00e 63b6f93a36
fdd91afd5c 3ff7167685
1191c86657 c2c0a23bdf
9b845d0a42 0693522015
1853d2754b 15dfaeb9dc
d850a0e2c0 0d1d5c4e52
368b634537 40677fe3d0
5d0ca1c213 6b9f361e63
a597a829a8 ead26c1e5a
8120268124 c21236cd3e
81d3839ac8 f78139a373
```
