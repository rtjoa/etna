From QuickChick Require Import QuickChick. Import QcNotation.
From Coq Require Import Bool ZArith List. Import ListNotations.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import MonadNotation.

From STLC Require Import Impl Spec.

Fixpoint genTyp (size : nat) (stack1 : nat) (stack2 : nat) : G (Typ) :=
  match size with
  | O  => 
    (* Frequency1 (single-branch) *) 
    (returnGen (TBool ))
  | S size1 => 
    (* Frequency2 *) (freq [
      (* TBool *) (match (size, stack1, stack2) with
      | (1, 2, 1) => 97
      | (1, 2, 4) => 97
      | (2, 0, 2) => 12
      | (2, 3, 2) => 0
      | (2, 5, 2) => 16
      | (2, 6, 2) => 6
      | _ => 500
      end,
      (returnGen (TBool ))); 
      (* TFun *) (match (size, stack1, stack2) with
      | (1, 2, 1) => 97
      | (1, 2, 4) => 97
      | (2, 0, 2) => 94
      | (2, 3, 2) => 97
      | (2, 5, 2) => 85
      | (2, 6, 2) => 94
      | _ => 500
      end,
      (bindGen (genTyp size1 stack2 1) 
      (fun p1 => 
        (bindGen (genTyp size1 stack2 4) 
        (fun p2 => 
          (returnGen (TFun p1 p2)))))))])
  end.

Fixpoint genExpr (size : nat) (stack1 : nat) (stack2 : nat) : G (Expr) :=
  match size with
  | O  => 
    (* Frequency3 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 91
      | (0, 3, 5) => 93
      | (0, 3, 6) => 96
      | (0, 5, 3) => 90
      | (0, 5, 5) => 82
      | (0, 5, 6) => 80
      | (0, 6, 3) => 91
      | (0, 6, 5) => 95
      | (0, 6, 6) => 96
      | _ => 500
      end,
      (bindGen 
      (* GenNat1 *)
      (let _weight_1 := match (size, stack1, stack2) with
      | (0, 3, 3) => 95
      | (0, 3, 5) => 84
      | (0, 3, 6) => 8
      | (0, 5, 3) => 7
      | (0, 5, 5) => 4
      | (0, 5, 6) => 65
      | (0, 6, 3) => 10
      | (0, 6, 5) => 90
      | (0, 6, 6) => 90
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_1, returnGen 1);
        (100-_weight_1, returnGen 0)
      ]) (fun n1 =>
      (let _weight_2 := match (size, stack1, stack2) with
      | (0, 3, 3) => 6
      | (0, 3, 5) => 12
      | (0, 3, 6) => 87
      | (0, 5, 3) => 7
      | (0, 5, 5) => 89
      | (0, 5, 6) => 5
      | (0, 6, 3) => 97
      | (0, 6, 5) => 92
      | (0, 6, 6) => 90
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_2, returnGen 2);
        (100-_weight_2, returnGen 0)
      ]) (fun n2 =>
      (let _weight_4 := match (size, stack1, stack2) with
      | (0, 3, 3) => 87
      | (0, 3, 5) => 11
      | (0, 3, 6) => 4
      | (0, 5, 3) => 94
      | (0, 5, 5) => 87
      | (0, 5, 6) => 3
      | (0, 6, 3) => 10
      | (0, 6, 5) => 9
      | (0, 6, 6) => 92
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_4, returnGen 4);
        (100-_weight_4, returnGen 0)
      ]) (fun n4 =>
      (let _weight_8 := match (size, stack1, stack2) with
      | (0, 3, 3) => 94
      | (0, 3, 5) => 87
      | (0, 3, 6) => 90
      | (0, 5, 3) => 72
      | (0, 5, 5) => 1
      | (0, 5, 6) => 25
      | (0, 6, 3) => 5
      | (0, 6, 5) => 97
      | (0, 6, 6) => 91
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_8, returnGen 8);
        (100-_weight_8, returnGen 0)
      ]) (fun n8 =>
      (let _weight_16 := match (size, stack1, stack2) with
      | (0, 3, 3) => 12
      | (0, 3, 5) => 14
      | (0, 3, 6) => 87
      | (0, 5, 3) => 37
      | (0, 5, 5) => 98
      | (0, 5, 6) => 7
      | (0, 6, 3) => 90
      | (0, 6, 5) => 11
      | (0, 6, 6) => 88
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_16, returnGen 16);
        (100-_weight_16, returnGen 0)
      ]) (fun n16 =>
      (let _weight_32 := match (size, stack1, stack2) with
      | (0, 3, 3) => 93
      | (0, 3, 5) => 90
      | (0, 3, 6) => 90
      | (0, 5, 3) => 62
      | (0, 5, 5) => 59
      | (0, 5, 6) => 48
      | (0, 6, 3) => 95
      | (0, 6, 5) => 11
      | (0, 6, 6) => 95
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_32, returnGen 32);
        (100-_weight_32, returnGen 0)
      ]) (fun n32 =>
        returnGen (n1 + n2 + n4 + n8 + n16 + n32)
      )))))))))))) 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 0
      | (0, 3, 5) => 18
      | (0, 3, 6) => 1
      | (0, 5, 3) => 0
      | (0, 5, 5) => 4
      | (0, 5, 6) => 45
      | (0, 6, 3) => 0
      | (0, 6, 5) => 2
      | (0, 6, 6) => 10
      | _ => 500
      end,
      (bindGen 
      (* GenBool1 *) (let _weight_true := match (size, stack1, stack2) with
      | (0, 3, 3) => 50
      | (0, 3, 5) => 64
      | (0, 3, 6) => 12
      | (0, 5, 3) => 50
      | (0, 5, 5) => 21
      | (0, 5, 6) => 6
      | (0, 6, 3) => 50
      | (0, 6, 5) => 3
      | (0, 6, 6) => 38
      | _ => 500
      end
      in
      freq [
        (_weight_true, returnGen true);
        (100-_weight_true, returnGen false)
      ]) 
      (fun p1 => 
        (returnGen (Bool p1)))))])
  | S size1 => 
    (* Frequency4 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 61
      | (1, 3, 5) => 1
      | (1, 3, 6) => 53
      | (1, 5, 3) => 72
      | (1, 5, 5) => 50
      | (1, 5, 6) => 58
      | (1, 6, 3) => 91
      | (1, 6, 5) => 2
      | (1, 6, 6) => 24
      | (2, 3, 3) => 0
      | (2, 3, 5) => 42
      | (2, 3, 6) => 0
      | (2, 5, 3) => 51
      | (2, 5, 5) => 45
      | (2, 5, 6) => 45
      | (2, 6, 3) => 0
      | (2, 6, 5) => 28
      | (2, 6, 6) => 0
      | (3, 3, 3) => 49
      | (3, 3, 5) => 49
      | (3, 3, 6) => 49
      | (3, 5, 3) => 0
      | (3, 5, 5) => 27
      | (3, 5, 6) => 0
      | (3, 6, 3) => 48
      | (3, 6, 5) => 50
      | (3, 6, 6) => 48
      | (4, 0, 3) => 29
      | (4, 0, 5) => 0
      | (4, 0, 6) => 30
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      (* GenNat2 *)
      (let _weight_1 := match (size, stack1, stack2) with
      | (1, 3, 3) => 28
      | (1, 3, 5) => 65
      | (1, 3, 6) => 45
      | (1, 5, 3) => 57
      | (1, 5, 5) => 77
      | (1, 5, 6) => 40
      | (1, 6, 3) => 4
      | (1, 6, 5) => 12
      | (1, 6, 6) => 48
      | (2, 3, 3) => 14
      | (2, 3, 5) => 44
      | (2, 3, 6) => 72
      | (2, 5, 3) => 52
      | (2, 5, 5) => 52
      | (2, 5, 6) => 54
      | (2, 6, 3) => 43
      | (2, 6, 5) => 45
      | (2, 6, 6) => 32
      | (3, 3, 3) => 50
      | (3, 3, 5) => 45
      | (3, 3, 6) => 50
      | (3, 5, 3) => 70
      | (3, 5, 5) => 59
      | (3, 5, 6) => 66
      | (3, 6, 3) => 50
      | (3, 6, 5) => 49
      | (3, 6, 6) => 50
      | (4, 0, 3) => 50
      | (4, 0, 5) => 53
      | (4, 0, 6) => 50
      | (5, 0, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_1, returnGen 1);
        (100-_weight_1, returnGen 0)
      ]) (fun n1 =>
      (let _weight_2 := match (size, stack1, stack2) with
      | (1, 3, 3) => 50
      | (1, 3, 5) => 85
      | (1, 3, 6) => 87
      | (1, 5, 3) => 83
      | (1, 5, 5) => 77
      | (1, 5, 6) => 50
      | (1, 6, 3) => 83
      | (1, 6, 5) => 78
      | (1, 6, 6) => 100
      | (2, 3, 3) => 47
      | (2, 3, 5) => 41
      | (2, 3, 6) => 8
      | (2, 5, 3) => 64
      | (2, 5, 5) => 47
      | (2, 5, 6) => 48
      | (2, 6, 3) => 15
      | (2, 6, 5) => 42
      | (2, 6, 6) => 64
      | (3, 3, 3) => 50
      | (3, 3, 5) => 50
      | (3, 3, 6) => 50
      | (3, 5, 3) => 61
      | (3, 5, 5) => 52
      | (3, 5, 6) => 34
      | (3, 6, 3) => 50
      | (3, 6, 5) => 51
      | (3, 6, 6) => 50
      | (4, 0, 3) => 50
      | (4, 0, 5) => 43
      | (4, 0, 6) => 50
      | (5, 0, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_2, returnGen 2);
        (100-_weight_2, returnGen 0)
      ]) (fun n2 =>
      (let _weight_4 := match (size, stack1, stack2) with
      | (1, 3, 3) => 86
      | (1, 3, 5) => 86
      | (1, 3, 6) => 84
      | (1, 5, 3) => 37
      | (1, 5, 5) => 49
      | (1, 5, 6) => 61
      | (1, 6, 3) => 66
      | (1, 6, 5) => 69
      | (1, 6, 6) => 2
      | (2, 3, 3) => 88
      | (2, 3, 5) => 22
      | (2, 3, 6) => 40
      | (2, 5, 3) => 50
      | (2, 5, 5) => 49
      | (2, 5, 6) => 44
      | (2, 6, 3) => 81
      | (2, 6, 5) => 66
      | (2, 6, 6) => 32
      | (3, 3, 3) => 50
      | (3, 3, 5) => 57
      | (3, 3, 6) => 50
      | (3, 5, 3) => 52
      | (3, 5, 5) => 35
      | (3, 5, 6) => 61
      | (3, 6, 3) => 50
      | (3, 6, 5) => 50
      | (3, 6, 6) => 50
      | (4, 0, 3) => 50
      | (4, 0, 5) => 51
      | (4, 0, 6) => 50
      | (5, 0, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_4, returnGen 4);
        (100-_weight_4, returnGen 0)
      ]) (fun n4 =>
      (let _weight_8 := match (size, stack1, stack2) with
      | (1, 3, 3) => 33
      | (1, 3, 5) => 51
      | (1, 3, 6) => 96
      | (1, 5, 3) => 69
      | (1, 5, 5) => 42
      | (1, 5, 6) => 53
      | (1, 6, 3) => 46
      | (1, 6, 5) => 51
      | (1, 6, 6) => 86
      | (2, 3, 3) => 79
      | (2, 3, 5) => 62
      | (2, 3, 6) => 22
      | (2, 5, 3) => 55
      | (2, 5, 5) => 40
      | (2, 5, 6) => 48
      | (2, 6, 3) => 77
      | (2, 6, 5) => 64
      | (2, 6, 6) => 67
      | (3, 3, 3) => 50
      | (3, 3, 5) => 51
      | (3, 3, 6) => 50
      | (3, 5, 3) => 54
      | (3, 5, 5) => 51
      | (3, 5, 6) => 66
      | (3, 6, 3) => 50
      | (3, 6, 5) => 49
      | (3, 6, 6) => 50
      | (4, 0, 3) => 50
      | (4, 0, 5) => 52
      | (4, 0, 6) => 50
      | (5, 0, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_8, returnGen 8);
        (100-_weight_8, returnGen 0)
      ]) (fun n8 =>
      (let _weight_16 := match (size, stack1, stack2) with
      | (1, 3, 3) => 8
      | (1, 3, 5) => 96
      | (1, 3, 6) => 39
      | (1, 5, 3) => 44
      | (1, 5, 5) => 66
      | (1, 5, 6) => 22
      | (1, 6, 3) => 13
      | (1, 6, 5) => 11
      | (1, 6, 6) => 26
      | (2, 3, 3) => 42
      | (2, 3, 5) => 46
      | (2, 3, 6) => 93
      | (2, 5, 3) => 54
      | (2, 5, 5) => 60
      | (2, 5, 6) => 56
      | (2, 6, 3) => 95
      | (2, 6, 5) => 36
      | (2, 6, 6) => 61
      | (3, 3, 3) => 50
      | (3, 3, 5) => 55
      | (3, 3, 6) => 50
      | (3, 5, 3) => 82
      | (3, 5, 5) => 43
      | (3, 5, 6) => 67
      | (3, 6, 3) => 50
      | (3, 6, 5) => 52
      | (3, 6, 6) => 50
      | (4, 0, 3) => 50
      | (4, 0, 5) => 51
      | (4, 0, 6) => 50
      | (5, 0, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_16, returnGen 16);
        (100-_weight_16, returnGen 0)
      ]) (fun n16 =>
      (let _weight_32 := match (size, stack1, stack2) with
      | (1, 3, 3) => 22
      | (1, 3, 5) => 49
      | (1, 3, 6) => 85
      | (1, 5, 3) => 34
      | (1, 5, 5) => 54
      | (1, 5, 6) => 73
      | (1, 6, 3) => 53
      | (1, 6, 5) => 8
      | (1, 6, 6) => 10
      | (2, 3, 3) => 7
      | (2, 3, 5) => 41
      | (2, 3, 6) => 86
      | (2, 5, 3) => 45
      | (2, 5, 5) => 48
      | (2, 5, 6) => 57
      | (2, 6, 3) => 89
      | (2, 6, 5) => 60
      | (2, 6, 6) => 92
      | (3, 3, 3) => 50
      | (3, 3, 5) => 51
      | (3, 3, 6) => 50
      | (3, 5, 3) => 67
      | (3, 5, 5) => 60
      | (3, 5, 6) => 46
      | (3, 6, 3) => 50
      | (3, 6, 5) => 45
      | (3, 6, 6) => 50
      | (4, 0, 3) => 50
      | (4, 0, 5) => 51
      | (4, 0, 6) => 50
      | (5, 0, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_32, returnGen 32);
        (100-_weight_32, returnGen 0)
      ]) (fun n32 =>
        returnGen (n1 + n2 + n4 + n8 + n16 + n32)
      )))))))))))) 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 0
      | (1, 3, 5) => 1
      | (1, 3, 6) => 0
      | (1, 5, 3) => 3
      | (1, 5, 5) => 33
      | (1, 5, 6) => 47
      | (1, 6, 3) => 0
      | (1, 6, 5) => 0
      | (1, 6, 6) => 0
      | (2, 3, 3) => 0
      | (2, 3, 5) => 10
      | (2, 3, 6) => 0
      | (2, 5, 3) => 12
      | (2, 5, 5) => 42
      | (2, 5, 6) => 42
      | (2, 6, 3) => 0
      | (2, 6, 5) => 6
      | (2, 6, 6) => 0
      | (3, 3, 3) => 49
      | (3, 3, 5) => 49
      | (3, 3, 6) => 49
      | (3, 5, 3) => 0
      | (3, 5, 5) => 13
      | (3, 5, 6) => 0
      | (3, 6, 3) => 48
      | (3, 6, 5) => 49
      | (3, 6, 6) => 54
      | (4, 0, 3) => 29
      | (4, 0, 5) => 0
      | (4, 0, 6) => 62
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      (* GenBool2 *) (let _weight_true := match (size, stack1, stack2) with
      | (1, 3, 3) => 50
      | (1, 3, 5) => 12
      | (1, 3, 6) => 5
      | (1, 5, 3) => 50
      | (1, 5, 5) => 37
      | (1, 5, 6) => 41
      | (1, 6, 3) => 50
      | (1, 6, 5) => 82
      | (1, 6, 6) => 79
      | (2, 3, 3) => 50
      | (2, 3, 5) => 56
      | (2, 3, 6) => 38
      | (2, 5, 3) => 50
      | (2, 5, 5) => 47
      | (2, 5, 6) => 41
      | (2, 6, 3) => 50
      | (2, 6, 5) => 43
      | (2, 6, 6) => 41
      | (3, 3, 3) => 50
      | (3, 3, 5) => 56
      | (3, 3, 6) => 49
      | (3, 5, 3) => 50
      | (3, 5, 5) => 51
      | (3, 5, 6) => 60
      | (3, 6, 3) => 50
      | (3, 6, 5) => 49
      | (3, 6, 6) => 55
      | (4, 0, 3) => 50
      | (4, 0, 5) => 46
      | (4, 0, 6) => 50
      | (5, 0, 0) => 52
      | _ => 500
      end
      in
      freq [
        (_weight_true, returnGen true);
        (100-_weight_true, returnGen false)
      ]) 
      (fun p1 => 
        (returnGen (Bool p1))))); 
      (* Abs *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 94
      | (1, 3, 5) => 57
      | (1, 3, 6) => 93
      | (1, 5, 3) => 73
      | (1, 5, 5) => 60
      | (1, 5, 6) => 55
      | (1, 6, 3) => 97
      | (1, 6, 5) => 1
      | (1, 6, 6) => 97
      | (2, 3, 3) => 24
      | (2, 3, 5) => 77
      | (2, 3, 6) => 3
      | (2, 5, 3) => 71
      | (2, 5, 5) => 57
      | (2, 5, 6) => 54
      | (2, 6, 3) => 3
      | (2, 6, 5) => 60
      | (2, 6, 6) => 3
      | (3, 3, 3) => 53
      | (3, 3, 5) => 55
      | (3, 3, 6) => 50
      | (3, 5, 3) => 0
      | (3, 5, 5) => 58
      | (3, 5, 6) => 0
      | (3, 6, 3) => 54
      | (3, 6, 5) => 50
      | (3, 6, 6) => 49
      | (4, 0, 3) => 79
      | (4, 0, 5) => 0
      | (4, 0, 6) => 62
      | (5, 0, 0) => 97
      | _ => 500
      end,
      (bindGen (genTyp 2 stack2 2) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 5) 
        (fun p2 => 
          (returnGen (Abs p1 p2))))))); 
      (* App *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 97
      | (1, 3, 5) => 94
      | (1, 3, 6) => 98
      | (1, 5, 3) => 45
      | (1, 5, 5) => 58
      | (1, 5, 6) => 42
      | (1, 6, 3) => 95
      | (1, 6, 5) => 95
      | (1, 6, 6) => 97
      | (2, 3, 3) => 97
      | (2, 3, 5) => 56
      | (2, 3, 6) => 97
      | (2, 5, 3) => 56
      | (2, 5, 5) => 55
      | (2, 5, 6) => 58
      | (2, 6, 3) => 97
      | (2, 6, 5) => 81
      | (2, 6, 6) => 97
      | (3, 3, 3) => 49
      | (3, 3, 5) => 46
      | (3, 3, 6) => 52
      | (3, 5, 3) => 97
      | (3, 5, 5) => 78
      | (3, 5, 6) => 96
      | (3, 6, 3) => 50
      | (3, 6, 5) => 51
      | (3, 6, 6) => 49
      | (4, 0, 3) => 34
      | (4, 0, 5) => 97
      | (4, 0, 6) => 38
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen (genExpr size1 stack2 3) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 6) 
        (fun p2 => 
          (returnGen (App p1 p2)))))))])
  end.

Definition gSized :=
  (genExpr 5 0 0).

Definition test_prop_SinglePreserve :=
forAll gSized (fun (e: Expr) =>
  prop_SinglePreserve e).

(*! QuickChick test_prop_SinglePreserve. *)

Definition test_prop_MultiPreserve :=
forAll gSized (fun (e: Expr) =>
  prop_MultiPreserve e).

(*! QuickChick test_prop_MultiPreserve. *)
          
