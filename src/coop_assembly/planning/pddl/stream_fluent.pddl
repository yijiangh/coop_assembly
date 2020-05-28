(define (stream construction)

   ; we can try the original incremental
   ; by commenting out the fluent
   ; uncomment the test-cfree stream function
   ; and (in the domain_fluent file) nSafeTraj in the place action's precondition
  (:stream sample-place
    :inputs (?r ?e)
    :domain (and (Robot ?r) (Element ?e) (Assigned ?r ?e))
    :fluents (Assembled)
    :outputs (?t)
    :certified (and
                    (PlaceAction ?r ?e ?t)
                    (Traj ?r ?t)
                )
  )

;   (:stream test-cfree
;     :inputs (?r ?t ?e)
;     :domain (and (Robot ?r) (Traj ?r ?t) (Element ?e))
;     :certified (CollisionFree ?r ?t ?e)
;   )

)
